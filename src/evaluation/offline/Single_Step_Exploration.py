import os
import sys
import json
import base64
import re
import asyncio
import aiofiles
from pathlib import Path
from tqdm.asyncio import tqdm_asyncio
from openai import AsyncOpenAI

Model_name = "Web-CogReasoner"
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[3])).resolve()

def resolve_asset_path(path):
    path = Path(path)
    if path.is_absolute():
        try:
            path = path.relative_to("/code/Web-CogReasoner")
        except ValueError:
            return str(path)
    return str(PROJECT_ROOT / path)

# Config
MODEL_NAME = "qwen2vl"
MAX_SAMPLE = 70
MAX_CONCURRENT_REQUESTS = 5
ACCURACY_PRINT_INTERVAL = 10
TEST_JSON_PATH = str(PROJECT_ROOT / "benchmark/Exploring/Single_Step_Exploration.json")
OUTPUT_JSON_PATH = str(PROJECT_ROOT / f"results_Web-CogBench/{Model_name}-Single_Step.json")

# OpenAI client
client = AsyncOpenAI(
    api_key=os.getenv("MODEL_API_KEY", "EMPTY"),
    base_url=os.getenv("MODEL_ENDPOINT", "http://localhost:8080/v1"),
)

def extract_action(text: str):
    if not text:
        return None
    type_match = re.search(r"Action:\s+type\s+\[(\d+)\]\s+\[(.*?)\]", text, re.IGNORECASE)
    if type_match:
        node_id = type_match.group(1)
        value = type_match.group(2)
        return f"TYPE({node_id}, {value})"
    simple_match = re.search(r"Action:\s+(\w+)\s+\[(\d+)\]", text, re.IGNORECASE)
    if simple_match:
        action = simple_match.group(1).upper()
        node_id = simple_match.group(2)
        return f"{action}({node_id})"
    return None

def parse_ground_truth(gt_content: str):
    action_parts = gt_content.split(';')
    parsed_actions = [extract_action(part.strip()) for part in action_parts]
    return [action for action in parsed_actions if action is not None]

def compare_actions(prediction: str, ground_truth_list: list) -> bool:
    """Compare predicted action to ground-truth list with TYPE node-id rule."""
    if not prediction:
        return False

    pred_match = re.match(r"(\w+)\((\d+)", prediction)
    if not pred_match:
        return False
    pred_action_type = pred_match.group(1)
    pred_node_id = pred_match.group(2)

    for gt_action in ground_truth_list:
        if prediction == gt_action:
            return True

        if pred_action_type == "TYPE" and gt_action.startswith("TYPE("):
            gt_match = re.match(r"TYPE\((\d+)", gt_action)
            if gt_match:
                gt_node_id = gt_match.group(1)
                if pred_node_id == gt_node_id:
                    return True
    
    return False

async def process_item(index, item, sem, stats):
    async with sem:
        image_paths = [resolve_asset_path(path) for path in item["images"]]
        prompt = item["messages"][0]["content"]
        
        gt_json_str = item["messages"][-1]["content"]
        gt_answers_list = parse_ground_truth(gt_json_str)

        if not gt_answers_list:
            print(f"Warning: Failed to parse ground truth: {gt_json_str}")
            return {
                "images": image_paths,
                "prompt": prompt,
                "ground_truth": "INVALID_GT_FORMAT",
                "prediction": None,
                "match": False,
                "raw_model_output": "Ground truth format is invalid."
            }

        image_contents = []
        for path in image_paths:
            try:
                async with aiofiles.open(path, "rb") as f:
                    content = await f.read()
                encoded_image = base64.b64encode(content).decode("utf-8")
                image_contents.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image;base64,{encoded_image}"}
                })
            except FileNotFoundError:
                error_msg = f"[ERROR] Image not found at {path}"
                print(error_msg)
                return {
                    "images": image_paths,
                    "prompt": prompt,
                    "ground_truth": ";".join(gt_answers_list),
                    "prediction": None,
                    "match": False,
                    "raw_model_output": error_msg
                }

        messages = [
            {
                "role": "user",
                "content": image_contents + [
                    {
                        "type": "text",
                        "text": prompt.strip(),
                    }
                ],
            },
        ]

        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.1,
                top_p=0.95,
                max_tokens=2048,
            )
            pred_text = response.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"Model inference failed for sample {index}: {e}") from e

        pred_answer = extract_action(pred_text)
        
        match = compare_actions(pred_answer, gt_answers_list)
        
        stats["total"] += 1
        stats["correct"] += int(match)

        if stats["total"] > 0 and stats["total"] % ACCURACY_PRINT_INTERVAL == 0:
            acc = stats["correct"] / stats["total"] * 100
            print(f"\nStep {stats['total']}: Accuracy = {acc:.2f}%\n")

        return {
            "images": image_paths,
            "prompt": prompt,
            "ground_truth": ";".join(gt_answers_list),
            "prediction": pred_answer,
            "match": match,
            "raw_model_output": pred_text
        }

async def main():
    with open(TEST_JSON_PATH, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    
    if MAX_SAMPLE is not None and MAX_SAMPLE > 0:
        test_data = test_data[:MAX_SAMPLE]

    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    stats = {"total": 0, "correct": 0}
    tasks = [process_item(i, item, sem, stats) for i, item in enumerate(test_data)]

    print(f"\nStarting evaluation of {len(tasks)} samples with model '{MODEL_NAME}'...\n")
    results = await tqdm_asyncio.gather(*tasks)
    
    valid_results = [r for r in results if r is not None]
    if not valid_results:
        print("\nNo valid samples were processed. Evaluation cannot be completed.")
        return

    final_total = stats["total"]
    final_correct = stats["correct"]
    accuracy = (final_correct / final_total * 100) if final_total > 0 else 0
    
    errors = [r for r in valid_results if not r["match"]]

    output = {
        "model_name": Model_name,
        "metrics": {
            "total_processed": final_total,
            "correct": final_correct,
            "accuracy": accuracy
        },
        "results": valid_results,
        "errors": errors
    }

    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nEvaluation Complete")
    print(f"Accuracy: {accuracy:.2f}% ({final_correct}/{final_total})")
    print(f"Results saved to: {OUTPUT_JSON_PATH}")

    if errors:
        print("\nSample Errors (up to 5):")
        for r in errors[:5]:
            print(f"- Images       : {', '.join(r['images'])}")
            print(f"  Ground Truth : {r['ground_truth']}")
            print(f"  Prediction   : {r['prediction']}")
            raw_output_snippet = r['raw_model_output'].replace('\n', ' ')
            if len(raw_output_snippet) > 200:
                raw_output_snippet = "..." + raw_output_snippet[-200:]
            print(f"  Raw Output   : {raw_output_snippet}\n")

    await client.aclose()

if __name__ == "__main__":
    asyncio.run(main())
