import os
import json
import base64
import pickle
import re
import uuid
import asyncio
import aiofiles
from PIL import Image
from tqdm.asyncio import tqdm_asyncio
import requests

Model_name = "OpenWebVoyager"
TEST_JSON_PATH = "/code/CogReasoner/Test/Popup_close.json"
MAX_SAMPLE = 60
MAX_CONCURRENT_REQUESTS = 5
ACCURACY_PRINT_INTERVAL = 10
OUTPUT_JSON_PATH = f"/code/CogReasoner/Code/Evalaute/Result/Test-{Model_name}-Popup_close.json"

SERVER_URL = "http://127.0.0.1:8080/predict"

def extract_action_from_output(text):
    match = re.search(r"Action:\s+(\w+)\s+\[(\d+)\]", text, re.IGNORECASE)
    if match:
        action = match.group(1).upper()
        node_id = match.group(2)
        return f"{action}({node_id})"
    return None

async def process_item(index, item, sem, stats):
    async with sem:
        image_paths = item["images"]
        prompt = item["messages"][0]["content"]
        prompt = prompt[prompt.find("OBSERVATION:"):]
        gt_json_str = item["messages"][-1]["content"]
        try:
            gt_data = json.loads(gt_json_str)
            action = gt_data.get('ACTION', gt_data.get('action', '')).upper()
            node_id = gt_data.get('NODE_ID', gt_data.get('node_id'))
            gt_answer = f"{action}({node_id})"
        except (json.JSONDecodeError, TypeError, AttributeError):
            print(f"警告: 无法解析 Ground Truth: {gt_json_str}")
            gt_answer = "INVALID_GT_FORMAT"

        image_list = []
        for path in image_paths:
            try:
                pil_image = Image.open(path).convert("RGB")
                image_list.append(pil_image)
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

        try:
            pickled_images = pickle.dumps(image_list)
            base64_encoded_images = base64.b64encode(pickled_images).decode("utf-8")
        except Exception as e:
            error_msg = f"[ERROR] Image encoding failed: {e}"
            print(error_msg)
            return {
                "images": image_paths,
                "ground_truth": gt_answer,
                "prediction": None,
                "match": False,
                "raw_model_output": error_msg
            }

        system_prompt = "You are a helpful assistant that analyzes UI screenshots and determines the next action."
        prompt_text = prompt.strip()
        user_content = "Determine the single, most direct action required to close or dismiss the popup." + prompt_text

        payload = {
            "id": str(uuid.uuid4()),
            "conversations": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "images": base64_encoded_images
        }

        try:
            response = requests.post(SERVER_URL, json=payload, timeout=120)
            if response.status_code == 200:
                result = response.json()
                pred_text = result.get("text", "").strip()
            else:
                pred_text = f"[HTTP_ERROR {response.status_code}] {response.text}"
        except Exception as e:
            pred_text = f"[NETWORK_ERROR] {str(e)}"

        pred_answer = extract_action_from_output(pred_text)
        match = pred_answer == gt_answer

        stats["total"] += 1
        stats["correct"] += int(match)

        if stats["total"] % ACCURACY_PRINT_INTERVAL == 0:
            acc = stats["correct"] / stats["total"] * 100
            print(f"\n📊 Step {stats['total']}: Accuracy = {acc:.2f}%\n")

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

    if MAX_SAMPLE:
        test_data = test_data[:MAX_SAMPLE]

    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    stats = {"total": 0, "correct": 0}
    tasks = [process_item(i, item, sem, stats) for i, item in enumerate(test_data)]

    print(f"\n🚀 Starting evaluation of {len(tasks)} samples with model '{Model_name}'...\n")
    results = await tqdm_asyncio.gather(*tasks)

    valid_results = [r for r in results if r is not None]
    if not valid_results:
        print("\n❌ No valid samples were processed. Evaluation cannot be completed.")
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

    print(f"\n✅ Evaluation Complete")
    print(f"🎯 Accuracy: {accuracy:.2f}% ({final_correct}/{final_total})")
    print(f"📁 Results saved to: {OUTPUT_JSON_PATH}")

    if errors:
        print("\n❌ Sample Errors (up to 5):")
        for r in errors[:5]:
            print(f"- Images       : {', '.join(r['images'])}")
            print(f"  Ground Truth : {r['ground_truth']}")
            print(f"  Prediction   : {r['prediction']}")
            raw_output_snippet = r['raw_model_output'][-200:].replace('\n', ' ')
            print(f"  Raw Output...: ...{raw_output_snippet}\n")

if __name__ == "__main__":
    asyncio.run(main())
