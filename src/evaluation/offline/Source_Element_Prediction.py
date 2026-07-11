import os
import json
import base64
import re
import asyncio
import aiofiles
import sys
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

# Configuration
TEST_JSON_PATH = str(PROJECT_ROOT / "benchmark/Memorizing/Source_Element_Prediction.json")  # Path to test dataset
MODEL_NAME = "qwen2vl"  # Target model
MAX_SAMPLE = 44  # Number of samples to test
MAX_CONCURRENT_REQUESTS = 5  # Concurrent request limit
ACCURACY_PRINT_INTERVAL = 10  # Logging interval
OUTPUT_JSON_PATH = str(PROJECT_ROOT / f"results_Web-CogBench/{Model_name}-Source_Element_Prediction.json")  # Output path

# Initialize OpenAI client (vLLM API)
client = AsyncOpenAI(
    api_key=os.getenv("MODEL_API_KEY", "EMPTY"),
    base_url=os.getenv("MODEL_ENDPOINT", "http://localhost:8080/v1"),
)

# Extract answer letter from model output
def extract_answer_letter(text):
    match = re.search(r"\b([A-Z])\b", text.strip(), re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None

# Asynchronous processing of a single item
async def process_item(index, item, sem, stats):
    async with sem:
        image_paths = [resolve_asset_path(path) for path in item["images"]]
        # Get all correct options from comma-separated string
        gt_answer_str = item["messages"][-1]["content"].strip().upper()
        possible_gt_answers = {opt.strip() for opt in gt_answer_str.split(',')}
        prompt = item["messages"][0]["content"]

        # Encode images to base64
        image_contents = []
        for path in image_paths:
            async with aiofiles.open(path, "rb") as f:
                content = await f.read()
            encoded_image = base64.b64encode(content).decode("utf-8")
            image_contents.append({
                "type": "image_url",
                "image_url": {"url": f"data:image;base64,{encoded_image}"}
            })

        # Construct messages
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
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
                max_tokens=512,
            )
            pred_text = response.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"Model inference failed for sample {index}: {e}") from e

        pred_answer = extract_answer_letter(pred_text)
        
        # Check if predicted answer is in correct options
        match = pred_answer is not None and pred_answer in possible_gt_answers

        stats["total"] += 1
        stats["correct"] += int(match)

        if stats["total"] % ACCURACY_PRINT_INTERVAL == 0:
            acc = stats["correct"] / stats["total"] * 100
            print(f"\n📊 Step {stats['total']}: Accuracy = {acc:.2f}%\n")

        return {
            "images": image_paths,
            "ground_truth": gt_answer_str,
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

    accuracy = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
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

    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Console summary
    print(f"\n✅ Evaluation Complete")
    print(f"🎯 Accuracy: {accuracy:.2f}%")
    print(f"📁 Results saved to: {OUTPUT_JSON_PATH}")

    # Display sample errors
    print("\n❌ Sample Errors (up to 5):")
    for r in errors[:5]:
        print(f"- Images       : {r['images']}")
        print(f"  Ground Truth : {r['ground_truth']}")
        print(f"  Prediction   : {r['prediction']}")
        print(f"  Raw Output   : {r['raw_model_output']}\n")

    await client.aclose()  # Release pool

# Run main
if __name__ == "__main__":
    asyncio.run(main())
    sys.exit(0)  # Force exit to ensure cleanup
