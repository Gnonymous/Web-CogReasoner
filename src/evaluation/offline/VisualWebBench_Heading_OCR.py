import os
import sys
import json
import base64
import asyncio
import aiofiles
from pathlib import Path
from tqdm.asyncio import tqdm_asyncio
from openai import AsyncOpenAI
from rouge import Rouge

Test_Model = "Web-CogReasoner"
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
TEST_JSON_PATH = str(PROJECT_ROOT / "benchmark/VisualWebBench/VisualWebBench_Heading_OCR_46.json")
OUTPUT_JSON_PATH = str(PROJECT_ROOT / f"results_VisualWebBench/{Test_Model}-VisualWebBench_HeadingOCR_46.json")
MAX_SAMPLE = 46
MAX_CONCURRENT_REQUESTS = 5
ACCURACY_PRINT_INTERVAL = 10
MODEL_NAME = "qwen2vl"
BASE_URL = os.getenv("MODEL_ENDPOINT", "http://localhost:8080/v1")

# OpenAI client
client = AsyncOpenAI(
    api_key=os.getenv("MODEL_API_KEY", "EMPTY"),
    base_url=BASE_URL,
)

FIXED_PROMPT = (
    "You are given a screenshot of a webpage. Please generate the main text within the screenshot, "
    "which can be regarded as the heading of the webpage.\n\n"
    "You should directly tell me the main content, and do not output any explanation or any other contents."
)

def eval_heading_ocr(preds, golds, **kwargs):
    assert len(preds) == len(golds)
    for i in range(len(preds)):
        if not preds[i]:
            preds[i] = " "
    rouge = Rouge(metrics=['rouge-1', 'rouge-2', 'rouge-l'])
    scores = rouge.get_scores(preds, golds, avg=True)
    return dict(
        rouge_1=scores['rouge-1']['f'] * 100,
        rouge_2=scores['rouge-2']['f'] * 100,
        rouge_l=scores['rouge-l']['f'] * 100
    )

async def process_item(index, item, sem):
    async with sem:
        image_path = resolve_asset_path(item["images"][0])
        ground_truth = item["messages"][1]["content"].strip()

        async with aiofiles.open(image_path, "rb") as f:
            content = await f.read()
        encoded_image = base64.b64encode(content).decode("utf-8")
        image_data_uri = f"data:image;base64,{encoded_image}"

        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_data_uri}},
                            {"type": "text", "text": FIXED_PROMPT},
                        ],
                    },
                ],
                temperature=0.1,
                top_p=0.95,
                max_tokens=512,
            )
            pred_text = response.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"Model inference failed for sample {index}: {e}") from e

        return {
            "image": image_path,
            "ground_truth": ground_truth,
            "prediction": pred_text,
        }

async def main():
    with open(TEST_JSON_PATH, "r", encoding="utf-8") as f:
        test_data = json.load(f)[:MAX_SAMPLE]

    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    tasks = [process_item(i, item, sem) for i, item in enumerate(test_data)]

    print(f"\n🚀 Starting evaluation on {len(tasks)} samples...\n")
    results = await tqdm_asyncio.gather(*tasks)

    predictions = [r["prediction"] for r in results]
    references = [r["ground_truth"] for r in results]

    metrics = eval_heading_ocr(predictions, references)

    output = {
        "metrics": metrics,
        "results": results,
    }

    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nEvaluation Complete")
    print(f"Metrics: {json.dumps(metrics, indent=2)}")
    print(f"Results saved at: {OUTPUT_JSON_PATH}")

    await client.close()

if __name__ == "__main__":
    asyncio.run(main())
    sys.exit(0)
