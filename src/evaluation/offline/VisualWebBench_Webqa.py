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
TEST_JSON_PATH = str(PROJECT_ROOT / "benchmark/VisualWebBench/VisualWebBench_webqa.json")
OUTPUT_JSON_PATH = str(PROJECT_ROOT / f"results_VisualWebBench/{Test_Model}-VisualWebBench_WebQA.json")
MAX_SAMPLE = 314
MAX_CONCURRENT_REQUESTS = 5
MODEL_NAME = "qwen2vl"
BASE_URL = os.getenv("MODEL_ENDPOINT", "http://localhost:8080/v1")

# OpenAI client
client = AsyncOpenAI(
    api_key=os.getenv("MODEL_API_KEY", "EMPTY"),
    base_url=BASE_URL,
)

def eval_webqa(preds, golds, **kwargs):
    """Compute WebQA F1."""
    assert len(preds) == len(golds), "Predictions and references must be the same length."
    f1_scores = []
    rouge = Rouge(metrics=['rouge-1'])
    for pred, gold_list in zip(preds, golds):
        if not pred:
            pred = " "
        
        try:
            current_f1 = max([rouge.get_scores([pred], [gold], avg=True)['rouge-1']['f'] for gold in gold_list])
            f1_scores.append(current_f1)
        except Exception as e:
            print(f"Warning: Could not compute F1 score for pred='{pred}' and gold_list='{gold_list}'. Error: {e}")
            f1_scores.append(0.0)

    if not f1_scores:
        return dict(f1=0.0)

    return dict(
        f1=sum(f1_scores) / len(f1_scores) * 100
    )

async def process_item(index, item, sem):
    async with sem:
        image_path = resolve_asset_path(item["images"][0])
        
        ground_truth = [item["messages"][1]["content"].strip()]
        
        user_prompt = item["messages"][0]["content"]

        async with aiofiles.open(image_path, "rb") as f:
            content = await f.read()
        encoded_image = base64.b64encode(content).decode("utf-8")
        image_data_uri = f"data:image;base64,{encoded_image}"

        try:
            prompt_text = user_prompt.replace("<image>\n", "").strip()

            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_data_uri}},
                            {"type": "text", "text": prompt_text},
                        ],
                    },
                ],
                temperature=0.1,
                top_p=0.95,
                max_tokens=1024,
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
    try:
        with open(TEST_JSON_PATH, "r", encoding="utf-8") as f:
            test_data = json.load(f)[:MAX_SAMPLE]
    except FileNotFoundError:
        print(f"Error: Test file not found at {TEST_JSON_PATH}")
        return

    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    tasks = [process_item(i, item, sem) for i, item in enumerate(test_data)]

    print(f"\nStarting evaluation for WebQA on {len(tasks)} samples...\n")
    results = await tqdm_asyncio.gather(*tasks)

    predictions = [r["prediction"] for r in results]
    references = [r["ground_truth"] for r in results]

    metrics = eval_webqa(predictions, references)

    output = {
        "task": "WebQA",
        "model": Test_Model,
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
    try:
        from rouge import Rouge
    except ImportError:
        print("Error: rouge is not installed. Run 'pip install rouge' or 'pip install rouge-chinese'.")
        sys.exit(1)
        
    asyncio.run(main())
    sys.exit(0)
