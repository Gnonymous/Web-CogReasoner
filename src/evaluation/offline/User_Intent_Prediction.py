import asyncio
import aiohttp
import google.generativeai as genai
import json
import os
import argparse
import base64
from tqdm.asyncio import tqdm_asyncio
from typing import Dict, Any, List
from datetime import datetime
import numpy as np
from pathlib import Path

# Config
Test_Model = "Web-CogReasoner"
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[3])).resolve()
VLLM_API_URL = os.getenv("MODEL_ENDPOINT", "http://localhost:8080/v1").rstrip("/") + "/chat/completions"
GEMINI_MODEL_NAME = 'gemini-2.5-pro'
MAX_CONCURRENT_REQUESTS = 2
Test_JSON_PATH = str(PROJECT_ROOT / "benchmark/Exploring/User_Intent_Prediction.json")
Inference_output_file = str(PROJECT_ROOT / f"results_Web-CogBench/Raw_Answer-{Test_Model}-User_Intent_Prediction.jsonl")
OUTPUT_JSON_PATH = str(PROJECT_ROOT / f"results_Web-CogBench/{Test_Model}-User_Intent_Prediction.json")

# Prompts
def get_gemini_evaluator_prompt(ground_truth: str, model_answer: str) -> str:
    """Evaluator prompt for user intent prediction."""
    return f"""You are a meticulous and impartial AI evaluator for a web navigation understanding benchmark. Your task is to assess the quality of a candidate model's predicted user intent by comparing it strictly against the Ground Truth Answer.

Your evaluation must be based *exclusively* on the information provided in the "Ground Truth Answer". Do not use any external knowledge or make assumptions beyond what is written in the ground truth.

Evaluate the candidate answer on the following three aspects:

- **Evidence Alignment**:  
  How well does the candidate use visual differences between the screenshots (layout changes, UI elements, visible filters, product count changes, etc.) to support the reasoning?

- **Intent Accuracy**:  
  Does the candidate correctly infer the user's goal or intent as described in the ground truth? Consider semantic equivalence and logical consistency.

- **Reasoning Quality**:  
  Does the candidate provide a coherent and step-wise explanation for how the user's intent evolves across the three screenshots?

---

**[Ground Truth Answer]**
{ground_truth}
---
**[Candidate Model's Answer]**
{model_answer}
---

**Evaluation Criteria & Scoring:**
- **Score 1:** Completely incorrect or missing.
- **Score 2:** Mostly incorrect, with a minor element of truth.
- **Score 3:** Partially correct, but misses significant details mentioned in the ground truth.
- **Score 4:** Mostly correct, with only minor inaccuracies or omissions compared to the ground truth.
- **Score 5:** Fully and accurately captures all relevant information present in the ground truth.

Your response MUST be a single, valid JSON object, adhering to the following structure. Do not add any text before or after the JSON object.

{{
  "evidence_alignment_score": <integer_score>,
  "evidence_alignment_justification": "<Your brief justification for the evidence score, referencing the ground truth>",
  "intent_accuracy_score": <integer_score>,
  "intent_accuracy_justification": "<Your brief justification for the intent score, referencing the ground truth>",
  "reasoning_quality_score": <integer_score>,
  "reasoning_quality_justification": "<Your brief justification for the reasoning score, referencing the ground truth>",
  "overall_score": <A final holistic integer score from 1 to 5, considering all aspects>,
  "overall_justification": "<A final summary of the model's performance on this example>"
}}
"""

def encode_image_to_base64(image_path: str) -> str:
    """Encode image to base64."""
    path = Path(image_path)
    if path.is_absolute():
        try:
            path = PROJECT_ROOT / path.relative_to("/code/Web-CogReasoner")
        except ValueError:
            pass
    else:
        path = PROJECT_ROOT / path
    image_path = str(path)
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"警告: 在路径 {image_path} 未找到图片文件")
        return None

def create_vllm_payload(user_prompt: str, image_base64_list: List[str]) -> Dict[str, Any]:
    """Create vLLM payload with multiple images."""
    content = [{"type": "text", "text": user_prompt}]
    for image_base64 in image_base64_list:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_base64}"}
        })

    return {
        "model": "qwen2vl",  # Must match the vLLM model name
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ],
        "max_tokens": 1024,
        "temperature": 0.1
    }

async def run_inference(item: Dict[str, Any], session: aiohttp.ClientSession, semaphore: asyncio.Semaphore) -> Dict[str, Any]:
    """Run inference and return the model answer."""
    async with semaphore:
        image_paths = item['images']
        image_id_base = os.path.basename(image_paths[0]) if image_paths else "no_image"
        
        user_prompt = item['messages'][0]['content']
        model_answer = None

        image_base64_list = [encode_image_to_base64(p) for p in image_paths]
        image_base64_list = [b64 for b64 in image_base64_list if b64 is not None]

        if not image_base64_list:
            model_answer = "Error: All image files not found or failed to encode."
        else:
            payload = create_vllm_payload(user_prompt, image_base64_list)
            try:
                async with session.post(VLLM_API_URL, json=payload, timeout=120) as response:
                    response.raise_for_status()
                    result = await response.json()
                    model_answer = result['choices'][0]['message']['content']
            except Exception as e:
                model_answer = f"Error during vLLM inference: {e}"
        return {"id": item.get("id", image_id_base), "model_answer": model_answer}

async def run_evaluation(item: Dict[str, Any], gemini_model: genai.GenerativeModel, semaphore: asyncio.Semaphore) -> Dict[str, Any]:
    """Run evaluation and attach results."""
    async with semaphore:
        ground_truth = item['messages'][1]['content']
        model_answer = item.get('model_answer', '')
        evaluation = None
        if "Error:" in model_answer or not model_answer:
            evaluation = {"error": "Skipped evaluation due to inference error or empty answer."}
        else:
            eval_prompt = get_gemini_evaluator_prompt(ground_truth, model_answer)
            try:
                response = await gemini_model.generate_content_async(
                    eval_prompt,
                    generation_config={
                        "response_mime_type": "application/json"
                    },
                )
                evaluation = json.loads(response.text)
            except Exception as e:
                evaluation = {"error": f"Error during Gemini evaluation: {e}"}
        item['evaluation'] = evaluation
        return item

def calculate_summary(results: List[Dict[str, Any]], model_name: str, benchmark_file: str, evaluator_model: str) -> Dict[str, Any]:
    """Compute summary statistics."""
    scores = {
        "evidence_alignment": [],
        "intent_accuracy": [],
        "reasoning_quality": [],
        "overall": []
    }
    
    successful_evals = 0
    failed_evals = 0

    for res in results:
        eval_data = res.get('evaluation', {})
        if 'error' in eval_data or not eval_data:
            failed_evals += 1
            continue
        
        successful_evals += 1
        scores["evidence_alignment"].append(eval_data.get("evidence_alignment_score", 0))
        scores["intent_accuracy"].append(eval_data.get("intent_accuracy_score", 0))
        scores["reasoning_quality"].append(eval_data.get("reasoning_quality_score", 0))
        scores["overall"].append(eval_data.get("overall_score", 0))

    average_scores = {
        "evidence_alignment_avg": round(np.mean(scores["evidence_alignment"]).item() if scores["evidence_alignment"] else 0, 3),
        "intent_accuracy_avg": round(np.mean(scores["intent_accuracy"]).item() if scores["intent_accuracy"] else 0, 3),
        "reasoning_quality_avg": round(np.mean(scores["reasoning_quality"]).item() if scores["reasoning_quality"] else 0, 3),
        "overall_avg": round(np.mean(scores["overall"]).item() if scores["overall"] else 0, 3)
    }

    summary = {
        "test_metadata": {
            "model_tested": model_name,
            "benchmark_file": os.path.basename(benchmark_file),
            "evaluator_model": evaluator_model,
            "test_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "evaluation_summary": {
            "total_samples": len(results),
            "successful_evaluations": successful_evals,
            "failed_evaluations": failed_evals,
            "average_scores": average_scores
        }
    }
    return summary

async def main():
    parser = argparse.ArgumentParser(description="Multi-stage benchmark tool: inference, evaluation, or all-in-one.")
    parser.add_argument("--gemini_api_key", default=os.getenv("GEMINI_API_KEY", ""), help="Google AI Studio API key.")
    parser.add_argument("--benchmark_file", default=Test_JSON_PATH, help="Path to the benchmark JSON file.")
    parser.add_argument("--output_file", default=OUTPUT_JSON_PATH, help="Path to save the final evaluation JSON.")
    parser.add_argument("--concurrency", type=int, default=MAX_CONCURRENT_REQUESTS, help="Max concurrent requests.")
    
    parser.add_argument("--inference_output_file", type=str, help="[Inference] Path to save .jsonl inference results. Defaults to preset path.")
    parser.add_argument("--evaluation_input_file", type=str, help="[Evaluation] Path to .jsonl file with model answers.")
    parser.add_argument("--mode", choices=['inference', 'evaluation', 'all'], help="Select mode: 'inference', 'evaluation', or 'all'.")
    
    args = parser.parse_args()

    if not args.mode:
        args.mode = 'all'

    if args.mode == 'inference':
        print("--- Entering [Inference Mode] ---")
        inference_output_path = args.inference_output_file if args.inference_output_file else Inference_output_file
        
        try:
            with open(args.benchmark_file, 'r', encoding='utf-8') as f:
                benchmark_items = json.load(f)
        except FileNotFoundError:
            print(f"Error: Benchmark file not found at {args.benchmark_file}.")
            return

        for i, item in enumerate(benchmark_items):
            if "id" not in item:
                img_path = item.get('images', [f'no_image_{i}'])[0]
                item["id"] = f"{os.path.basename(img_path)}_{i}"
        
        semaphore = asyncio.Semaphore(args.concurrency)
        
        async with aiohttp.ClientSession() as session:
            inference_tasks = [run_inference(item, session, semaphore) for item in benchmark_items]
            inference_results = await tqdm_asyncio.gather(*inference_tasks, desc="Inferring")

        output_dir = os.path.dirname(inference_output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with open(inference_output_path, 'w', encoding='utf-8') as f:
            for result in inference_results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        print(f"\nInference complete! Results saved to: {inference_output_path}")
        
    elif args.mode == 'evaluation':
        print("--- Entering [Evaluation Mode] ---")
        evaluation_input_path = args.evaluation_input_file if args.evaluation_input_file else Inference_output_file

        if not args.gemini_api_key or args.gemini_api_key == "YOUR_GEMINI_API_KEY":
            print("Error: Gemini API key is required for evaluation mode. Use --gemini_api_key.")
            return
            
        try:
            with open(args.benchmark_file, 'r', encoding='utf-8') as f:
                benchmark_data_list = json.load(f)
                benchmark_data_map = {}
                for i, item in enumerate(benchmark_data_list):
                    img_path = item.get('images', [f'no_image_{i}'])[0]
                    item_id = item.get("id", f"{os.path.basename(img_path)}_{i}")
                    benchmark_data_map[item_id] = item

            with open(evaluation_input_path, 'r', encoding='utf-8') as f:
                model_answers = [json.loads(line) for line in f]
        except FileNotFoundError as e:
            print(f"Error: Could not find input file - {e}")
            return
        
        items_to_evaluate = []
        for answer in model_answers:
            item_id = answer.get("id")
            if item_id in benchmark_data_map:
                full_item = benchmark_data_map[item_id]
                full_item['model_answer'] = answer['model_answer']
                items_to_evaluate.append(full_item)
            else:
                print(f"Warning: Item with ID '{item_id}' not found in benchmark data, skipping.")
        
        if not items_to_evaluate:
            print("Error: No data available for evaluation. Check ID matching.")
            return
            
        semaphore = asyncio.Semaphore(args.concurrency)
        
        genai.configure(api_key=args.gemini_api_key)
        gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        
        evaluation_tasks = [run_evaluation(item, gemini_model, semaphore) for item in items_to_evaluate]
        final_results_list = await tqdm_asyncio.gather(*evaluation_tasks, desc="Evaluating")

        summary = calculate_summary(
            results=final_results_list,
            model_name=Test_Model,
            benchmark_file=args.benchmark_file,
            evaluator_model=GEMINI_MODEL_NAME
        )

        final_output_object = {
            "summary": summary,
            "results": final_results_list
        }

        output_dir = os.path.dirname(args.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(final_output_object, f, indent=2, ensure_ascii=False)
        
        print("\n--- Evaluation Complete! Summary: ---")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print("--------------------------")
        print(f"\nFull results saved to: {args.output_file}")

    elif args.mode == 'all':
        print("--- Entering [Inference Mode] ---")
        inference_output_path = args.inference_output_file if args.inference_output_file else Inference_output_file

        try:
            with open(args.benchmark_file, 'r', encoding='utf-8') as f:
                benchmark_items = json.load(f)
        except FileNotFoundError:
            print(f"Error: Benchmark file not found at {args.benchmark_file}.")
            return
        except Exception as e:
            print(f"Error loading benchmark file: {e}")
            return

        for i, item in enumerate(benchmark_items):
            if "id" not in item:
                img_path = item.get('images', [f'no_image_{i}'])[0]
                item["id"] = f"{os.path.basename(img_path)}_{i}"

        semaphore = asyncio.Semaphore(args.concurrency)

        async with aiohttp.ClientSession() as session:
            inference_tasks = [run_inference(item, session, semaphore) for item in benchmark_items]
            inference_results = await tqdm_asyncio.gather(*inference_tasks, desc="Inferring")

        output_dir = os.path.dirname(inference_output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(inference_output_path, 'w', encoding='utf-8') as f:
            for result in inference_results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

        print(f"\nInference complete! Results saved to: {inference_output_path}")

        print("--- Entering [Evaluation Mode] ---")
        evaluation_input_path = args.evaluation_input_file if args.evaluation_input_file else inference_output_path

        if not args.gemini_api_key or args.gemini_api_key == "YOUR_GEMINI_API_KEY":
            print("Error: Gemini API key is required for evaluation mode. Use --gemini_api_key.")
            return

        try:
            with open(args.benchmark_file, 'r', encoding='utf-8') as f:
                benchmark_data_list = json.load(f)
                benchmark_data_map = {}
                for i, item in enumerate(benchmark_data_list):
                    img_path = item.get('images', [f'no_image_{i}'])[0]
                    item_id = item.get("id", f"{os.path.basename(img_path)}_{i}")
                    benchmark_data_map[item_id] = item

            with open(evaluation_input_path, 'r', encoding='utf-8') as f:
                model_answers = [json.loads(line) for line in f]
        except FileNotFoundError as e:
            print(f"Error: Could not find input file - {e}")
            return

        items_to_evaluate = []
        for answer in model_answers:
            item_id = answer.get("id")
            if item_id in benchmark_data_map:
                full_item = benchmark_data_map[item_id]
                full_item['model_answer'] = answer['model_answer']
                items_to_evaluate.append(full_item)
            else:
                print(f"Warning: Item with ID '{item_id}' not found in benchmark data, skipping.")

        if not items_to_evaluate:
            print("Error: No data available for evaluation. Check ID matching.")
            return

        semaphore = asyncio.Semaphore(args.concurrency)

        genai.configure(api_key=args.gemini_api_key)
        gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)

        evaluation_tasks = [run_evaluation(item, gemini_model, semaphore) for item in items_to_evaluate]
        final_results_list = await tqdm_asyncio.gather(*evaluation_tasks, desc="Evaluating")

        summary = calculate_summary(
            results=final_results_list,
            model_name=Test_Model,
            benchmark_file=args.benchmark_file,
            evaluator_model=GEMINI_MODEL_NAME
        )

        final_output_object = {
            "summary": summary,
            "results": final_results_list
        }

        output_dir = os.path.dirname(args.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(final_output_object, f, indent=2, ensure_ascii=False)

        print("\n--- Evaluation Complete! Summary: ---")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print("--------------------------")
        print(f"\nFull results saved to: {args.output_file}")

    else:
        print("Error: Ambiguous mode. Use --mode 'inference', 'evaluation', or 'all'.")

if __name__ == "__main__":
    asyncio.run(main())
