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
MAX_SAMPLE = 60
MAX_CONCURRENT_REQUESTS = 5
ACCURACY_PRINT_INTERVAL = 10
TEST_JSON_PATH = str(PROJECT_ROOT / "benchmark/Exploring/Popup_Close.json")
OUTPUT_JSON_PATH = str(PROJECT_ROOT / f"results_Web-CogBench/{Model_name}-Popup_close.json")

# OpenAI client
client = AsyncOpenAI(
    api_key=os.getenv("MODEL_API_KEY", "EMPTY"),
    base_url=os.getenv("MODEL_ENDPOINT", "http://localhost:8080/v1"),
)

def extract_action_from_output(text: str):
    """Extract action in format Action: action [node_id]."""
    match = re.search(r"Action:\s+(\w+)\s+\[(\d+)\]", text, re.IGNORECASE)
    
    if match:
        action = match.group(1).upper()
        node_id = match.group(2)
        return f"{action}({node_id})"
        
    return None

async def process_item(index, item, sem, stats):
    async with sem:
        image_paths = [resolve_asset_path(path) for path in item["images"]]
        prompt = item["messages"][0]["content"]
        prompt = prompt[prompt.find("OBSERVATION:"):]
        gt_json_str = item["messages"][-1]["content"]
        try:
            gt_data = json.loads(gt_json_str) 
            action = gt_data.get('ACTION', gt_data.get('action', '')).upper()
            node_id = gt_data.get('NODE_ID', gt_data.get('node_id'))
            gt_answer = f"{action}({node_id})"
        except (json.JSONDecodeError, TypeError, AttributeError):
            print(f"Warning: Failed to parse ground truth: {gt_json_str}")
            gt_answer = "INVALID_GT_FORMAT"

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
                    "ground_truth": gt_answer,
                    "prediction": None,
                    "match": False,
                    "raw_model_output": error_msg
                }

        messages = [
            {"role": "system", "content": "You are a helpful assistant that analyzes UI screenshots and determines the next action."},
            {
                "role": "user",
                "content": image_contents + [
                    {
                        "type": "text",
                        "text": "Determine the single, most direct action required to close or dismiss the popup." + prompt.strip(),
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

        pred_answer = extract_action_from_output(pred_text)
        
        match = pred_answer == gt_answer

        stats["total"] += 1
        stats["correct"] += int(match)

        if stats["total"] > 0 and stats["total"] % ACCURACY_PRINT_INTERVAL == 0:
            acc = stats["correct"] / stats["total"] * 100
            print(f"\nStep {stats['total']}: Accuracy = {acc:.2f}%\n")

        return {
            "images": image_paths,
            "ground_truth": gt_answer,
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
            raw_output_snippet = r['raw_model_output'][-200:].replace('\n', ' ')
            print(f"  Raw Output...: ...{raw_output_snippet}\n")

    await client.aclose()

if __name__ == "__main__":
    asyncio.run(main())
