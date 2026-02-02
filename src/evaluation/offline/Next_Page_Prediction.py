import os
import sys
import json
import base64
import re
import asyncio
import aiofiles
from tqdm.asyncio import tqdm_asyncio
from openai import AsyncOpenAI

Model_name = "Web-CogReasoner"

# Configuration
TEST_JSON_PATH = "/code/Web-CogReasoner/benchmark/Memorizing/Next_Page_Prediction_100.json"  # Path to test dataset
MODEL_NAME = "qwen2vl"  # Target model
MAX_SAMPLE = 100  # Number of samples to test
MAX_CONCURRENT_REQUESTS = 10  # Concurrent request limit
ACCURACY_PRINT_INTERVAL = 10  # Logging interval
OUTPUT_JSON_PATH = f"/code/Web-CogReasoner/results_Web-CogBench/{Model_name}-Next_Page_Prediction_100.json"  # Output path

# Initialize OpenAI client (vLLM API)
client = AsyncOpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8080/v1",
)

# Extract answer letter (A, B, C, etc.) from text
def extract_answer_letter(text):
    match = re.search(r"\b([A-H])\b", text.strip(), re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None

# Asynchronous processing of a single item
async def process_item(index, item, sem, stats):
    async with sem:
        image_path = item["images"][0]
        gt_answer = item["messages"][-1]["content"].strip().upper()
        prompt = item["messages"][0]["content"]

        # Encode image to base64
        async with aiofiles.open(image_path, "rb") as f:
            content = await f.read()
        encoded_image = base64.b64encode(content).decode("utf-8")
        image_data_uri = f"data:image;base64,{encoded_image}"

        try:
            # Model inference
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_data_uri}},
                            {
                                "type": "text",
                                "text": prompt + "Directly give the answer letter (A, B, C, D, E, F, G, H) without any explanation.",
                            },
                        ],
                    },
                ],
                temperature=0.1,
                top_p=0.95,
                max_tokens=512,
            )
            pred_text = response.choices[0].message.content.strip()
        except Exception as e:
            pred_text = f"[ERROR] {str(e)}"

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

# Main execution logic
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

    # Save results
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

    # Console output
    print(f"\n✅ Evaluation Complete")
    print(f"🎯 Accuracy: {accuracy:.2f}%")
    print(f"📁 Results saved to: {OUTPUT_JSON_PATH}")

    print("\n❌ Sample Errors (up to 5):")
    for r in errors[:5]:
        print(f"- Image        : {r['image']}")
        print(f"  Ground Truth : {r['ground_truth']}")
        print(f"  Prediction   : {r['prediction']}")
        print(f"  Raw Output   : {r['raw_model_output']}\n")

    await client.aclose()  # Release connection pool

# Run main
if __name__ == "__main__":
    asyncio.run(main())
    sys.exit(0)  # Force exit to ensure cleanup
