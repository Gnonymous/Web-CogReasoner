import os
import json
import base64
import pickle
import re
import uuid
import asyncio
import aiofiles
from tqdm.asyncio import tqdm_asyncio
from PIL import Image
import requests

Model_name = "OpenWebVoyager"
TEST_JSON_PATH = "/code/CogReasoner/Test/Next_Page_Prediction_100.json"
MAX_SAMPLE = 93
MAX_CONCURRENT_REQUESTS = 10
ACCURACY_PRINT_INTERVAL = 10
OUTPUT_JSON_PATH = f"/code/CogReasoner/Code/Evalaute/Result/Test-{Model_name}-Next_Page_Prediction_100.json"

SERVER_URL = "http://127.0.0.1:8080/predict"

def extract_answer_letter(text):
    match = re.search(r"\b([A-H])\b", text.strip(), re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None

async def process_item(index, item, sem, stats):
    async with sem:
        image_path = item["images"][0]
        gt_answer = item["messages"][-1]["content"].strip().upper()
        prompt = item["messages"][0]["content"]

        # 异步加载图片并处理为base64(pickle)
        try:
            async with aiofiles.open(image_path, "rb") as f:
                raw_bytes = await f.read()
            image = Image.open(image_path).convert("RGB")  # 强制为RGB
            pickled_images = pickle.dumps([image])
            base64_encoded_images = base64.b64encode(pickled_images).decode("utf-8")
        except Exception as e:
            print(f"❌ 图像处理错误 @ {image_path}: {e}")
            return {
                "image": image_path,
                "ground_truth": gt_answer,
                "prediction": None,
                "match": False,
                "raw_model_output": f"[ERROR] {str(e)}"
            }

        # 构造请求体
        prompt_text = " Thought this task and finally, directly give the answer letter (A, B, C, D, E, F, G, H) without any explanation." + prompt
        user_content = f"{prompt_text}"

        payload = {
            "id": str(uuid.uuid4()),
            "conversations": [{"role": "user", "content": user_content}],
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

        pred_answer = extract_answer_letter(pred_text)
        match = pred_answer == gt_answer

        stats["total"] += 1
        stats["correct"] += int(match)

        if stats["total"] % ACCURACY_PRINT_INTERVAL == 0:
            acc = stats["correct"] / stats["total"] * 100
            print(f"\n📊 Step {stats['total']}: Accuracy = {acc:.2f}%\n")

        return {
            "image": image_path,
            "ground_truth": gt_answer,
            "prediction": pred_answer,
            "match": match,
            "raw_model_output": pred_text
        }

async def main():
    with open(TEST_JSON_PATH, "r", encoding="utf-8") as f:
        test_data = json.load(f)[:MAX_SAMPLE]

    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    stats = {"total": 0, "correct": 0}
    tasks = [process_item(i, item, sem, stats) for i, item in enumerate(test_data)]

    print(f"\n🚀 Starting evaluation of {len(tasks)} samples...\n")
    results = await tqdm_asyncio.gather(*tasks)

    accuracy = stats["correct"] / stats["total"] * 100
    errors = [r for r in results if not r["match"]]

    output = {
        "metrics": {
            "total": stats["total"],
            "correct": stats["correct"],
            "accuracy": accuracy
        },
        "errors": errors
    }

    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Evaluation Complete")
    print(f"🎯 Accuracy: {accuracy:.2f}%")
    print(f"📁 Results saved to: {OUTPUT_JSON_PATH}")

    print("\n❌ Sample Errors (up to 5):")
    for r in errors[:5]:
        print(f"- Image        : {r['image']}")
        print(f"  Ground Truth : {r['ground_truth']}")
        print(f"  Prediction   : {r['prediction']}")
        print(f"  Raw Output   : {r['raw_model_output']}\n")

if __name__ == "__main__":
    asyncio.run(main())
