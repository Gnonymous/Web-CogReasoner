"""
Action Prediction Evaluation
Evaluates model's ability to predict the correct action from multiple choices.
"""
import os
import sys
import json
import base64
import re
import asyncio
import argparse
import aiofiles
from tqdm.asyncio import tqdm_asyncio
from openai import AsyncOpenAI


def extract_answer_letter(text):
    """Extract the first uppercase letter answer from model output."""
    match = re.search(r"\b([A-Z])\b", text.strip(), re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None


async def process_item(index, item, sem, client, model_name, stats):
    """Process a single evaluation sample."""
    async with sem:
        image_paths = item["images"]
        # Get ground truth, which may contain multiple correct options like "H,G,I,E"
        gt_answer_str = item["messages"][-1]["content"].strip().upper()
        # Create set of all correct options
        possible_gt_answers = {opt.strip() for opt in gt_answer_str.split(',')}
        prompt = item["messages"][0]["content"]

        # Encode all images to base64
        image_contents = []
        for path in image_paths:
            try:
                async with aiofiles.open(path, "rb") as f:
                    content = await f.read()
                encoded_image = base64.b64encode(content).decode("utf-8")
                image_contents.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded_image}"}
                })
            except FileNotFoundError:
                return {"match": False, "raw_model_output": f"Image not found: {path}"}

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": image_contents + [
                    {
                        "type": "text",
                        "text": prompt.strip() + " You should directly tell me your choice in a single uppercase letter.",
                    }
                ],
            },
        ]

        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.1,
                top_p=0.95,
                max_tokens=512,
            )
            pred_text = response.choices[0].message.content.strip()
        except Exception as e:
            pred_text = f"[ERROR] {str(e)}"

        pred_answer = extract_answer_letter(pred_text)
        # Check if prediction is in the set of correct answers
        match = pred_answer is not None and pred_answer in possible_gt_answers

        stats["total"] += 1
        stats["correct"] += int(match)

        if stats["total"] % 10 == 0:
            acc = stats["correct"] / stats["total"] * 100
            print(f"\nStep {stats['total']}: Accuracy = {acc:.2f}%\n")

        return {
            "images": image_paths,
            "ground_truth": gt_answer_str,
            "prediction": pred_answer,
            "match": match,
            "raw_model_output": pred_text
        }


async def main():
    parser = argparse.ArgumentParser(description="Action Prediction Evaluation")
    parser.add_argument("--model-endpoint", default="http://localhost:8080/v1",
                        help="vLLM or compatible API endpoint")
    parser.add_argument("--model-name", default="qwen2vl", help="Model name")
    parser.add_argument("--test-file", required=True, help="Path to test JSON file")
    parser.add_argument("--output-dir", default="./results", help="Output directory")
    parser.add_argument("--max-samples", type=int, default=0, help="Max samples (0=all)")
    parser.add_argument("--max-concurrent", type=int, default=5, help="Max concurrent requests")
    args = parser.parse_args()

    # Initialize client
    client = AsyncOpenAI(api_key="EMPTY", base_url=args.model_endpoint)

    # Load test data
    with open(args.test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    if args.max_samples > 0:
        test_data = test_data[:args.max_samples]

    sem = asyncio.Semaphore(args.max_concurrent)
    stats = {"total": 0, "correct": 0}
    tasks = [
        process_item(i, item, sem, client, args.model_name, stats) 
        for i, item in enumerate(test_data)
    ]

    print(f"\nStarting evaluation of {len(tasks)} samples...\n")
    results = await tqdm_asyncio.gather(*tasks)

    accuracy = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
    errors = [r for r in results if r and not r["match"]]

    output = {
        "metrics": {
            "total": stats["total"],
            "correct": stats["correct"],
            "accuracy": accuracy
        },
        "errors": errors[:50]  # Limit errors for output size
    }

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "action_prediction_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nEvaluation Complete")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Results saved to: {output_path}")

    if errors:
        print("\nSample Errors (up to 5):")
        for r in errors[:5]:
            print(f"- Ground Truth: {r['ground_truth']}, Prediction: {r['prediction']}")

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
    sys.exit(0)
