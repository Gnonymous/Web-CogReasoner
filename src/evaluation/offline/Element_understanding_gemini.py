import asyncio
import aiohttp
import google.generativeai as genai
import json
import os
import argparse
import base64
from tqdm.asyncio import tqdm_asyncio
from typing import Dict, Any, List
from openai import AsyncOpenAI
from datetime import datetime
import numpy as np # 引入numpy用于更安全地计算平均值

# --- 配置项 ---
Test_Model = "Gemini"
# 注意：使用f-string在这里定义全局变量可能不是最佳实践，但在脚本顶部可以接受
OUTPUT_JSON_PATH = f"/code/CogReasoner/Code/Evalaute/Result/Test-{Test_Model}-Element_Understanding_200.json"
Inference_output_file = f"/code/CogReasoner/Code/Evalaute/Result/Raw_Answer-{Test_Model}-Element_Understanding_200.jsonl"
VLLM_API_URL = "http://localhost:8080/v1/chat/completions"
GEMINI_MODEL_NAME = 'gemini-2.5-flash-lite-preview-06-17'
MAX_CONCURRENT_REQUESTS = 5 # 控制并发请求数，可根据您的硬件和API限制调整
client = AsyncOpenAI(api_key=os.getenv("GEMINI_API_KEY", ""),
                 base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
# --- 提示模板 ---
def get_gemini_evaluator_prompt(ground_truth: str, model_answer: str) -> str:
    """为Gemini评估器创建详细的提示。"""
    return f"""You are a meticulous and impartial AI evaluator for a web UI understanding benchmark. Your task is to assess the quality of a candidate model's answer by comparing it strictly against a ground truth reference.

Your evaluation must be based *exclusively* on the information provided in the "Ground Truth Answer". Do not use any external knowledge or make assumptions beyond what is written in the ground truth.

Evaluate the candidate answer on three specific aspects: Appearance, Position, and Function.

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
  "appearance_score": <integer_score>,
  "appearance_justification": "<Your brief justification for the appearance score, referencing the ground truth>",
  "position_score": <integer_score>,
  "position_justification": "<Your brief justification for the position score, referencing the ground truth>",
  "function_score": <integer_score>,
  "function_justification": "<Your brief justification for the function score, referencing the ground truth>",
  "overall_score": <A final holistic integer score from 1 to 5, considering all aspects>,
  "overall_justification": "<A final summary of the model's performance on this example>"
}}
"""

# --- 辅助函数 ---
def encode_image_to_base64(image_path: str) -> str:
    """将图片文件编码为base64字符串。"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"警告: 在路径 {image_path} 未找到图片文件")
        return None

def create_vllm_payload(user_prompt: str, image_base64: str) -> Dict[str, Any]:
    """为vLLM的OpenAI兼容API创建JSON负载。"""
    return {
        "model": "gemini-2.5-pro", # !!重要!! 确保这是您在vLLM中加载的模型名
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"}
                    }
                ]
            }
        ],
        "max_tokens": 1024,
        "temperature": 0.1
    }

# --- 核心异步函数 ---
async def run_inference(item: Dict[str, Any], session: aiohttp.ClientSession, semaphore: asyncio.Semaphore) -> Dict[str, Any]:
    """仅执行推理阶段，并返回包含答案的关键信息。"""
    async with semaphore:
        image_path = item['images'][0]
        user_prompt = item['messages'][0]['content']
        model_answer = None
        image_base64 = encode_image_to_base64(image_path)
        if not image_base64:
            model_answer = "Error: Image file not found."
        else:
            # payload = create_vllm_payload(user_prompt, image_base64)
            try:
            # 模型推理
                response = await client.chat.completions.create(
                model="gemini-2.5-pro",
                messages= [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"}
                    }
                ]
            }
        ],
                temperature=0.1,
                top_p=0.95,
                max_tokens=2048,
            )
                model_answer = response.choices[0].message.content
            except Exception as e:
                model_answer = f"Error during vLLM inference: {e}"
        # 只返回包含模型答案的关键信息，用于写入jsonl
        return {"id": item.get("id", os.path.basename(image_path)), "model_answer": model_answer}

async def run_evaluation(item: Dict[str, Any], gemini_model: genai.GenerativeModel, semaphore: asyncio.Semaphore) -> Dict[str, Any]:
    """仅执行评估阶段，并将评估结果添加到item字典中。"""
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

# --- 新增的辅助函数，用于计算摘要 ---
def calculate_summary(results: List[Dict[str, Any]], model_name: str, benchmark_file: str, evaluator_model: str) -> Dict[str, Any]:
    """计算评估结果的摘要统计信息。"""
    scores = {
        "appearance": [],
        "position": [],
        "function": [],
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
        scores["appearance"].append(eval_data.get("appearance_score", 0))
        scores["position"].append(eval_data.get("position_score", 0))
        scores["function"].append(eval_data.get("function_score", 0))
        scores["overall"].append(eval_data.get("overall_score", 0))

    # 使用numpy.mean来安全地处理空列表（如果所有评估都失败）
    average_scores = {
        "appearance_avg": round(np.mean(scores["appearance"]).item() if scores["appearance"] else 0, 3),
        "position_avg": round(np.mean(scores["position"]).item() if scores["position"] else 0, 3),
        "function_avg": round(np.mean(scores["function"]).item() if scores["function"] else 0, 3),
        "overall_avg": round(np.mean(scores["overall"]).item() if scores["overall"] else 0, 3)
    }

    summary = {
        "test_metadata": {
            "model_tested": model_name,
            "benchmark_file": os.path.basename(benchmark_file), # 只显示文件名，更简洁
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

# --- 主程序入口 ---
async def main():
    parser = argparse.ArgumentParser(description="分阶段Benchmark工具：可独立进行推理或评估。")
    parser.add_argument("--gemini_api_key", default=os.getenv("GEMINI_API_KEY", ""), help="您的Google AI Studio API密钥。")
    parser.add_argument("--benchmark_file", default="/code/CogReasoner/Test/Element_Understanding_sampled_200_clean.json", help="包含测试数据的JSON文件路径。")
    parser.add_argument("--output_file", default=OUTPUT_JSON_PATH, help="保存最终评估结果的JSON文件路径。")
    parser.add_argument("--concurrency", type=int, default=MAX_CONCURRENT_REQUESTS, help="最大并发请求数。")
    
    # 控制行为的参数
    parser.add_argument("--inference_output_file", type=str, help="[推理模式] 推理结果要保存到的.jsonl文件路径。如果未提供，将使用默认路径。")
    parser.add_argument("--evaluation_input_file", type=str, help="[评估模式] 包含模型答案的.jsonl文件路径。")
    parser.add_argument("--mode", choices=['inference', 'evaluation'], help="明确选择脚本运行模式：'inference' 或 'evaluation'。")
    
    args = parser.parse_args()

    # 如果没有明确模式，根据文件参数推断
    if not args.mode:
        if args.evaluation_input_file:
            args.mode = 'evaluation'
        else:
            args.mode = 'inference'

    # --- 模式选择 ---
    if args.mode == 'inference':
        # --- 推理模式 ---
        print("--- 进入 [推理模式] ---")
        inference_output_path = args.inference_output_file if args.inference_output_file else Inference_output_file
        
        try:
            with open(args.benchmark_file, 'r', encoding='utf-8') as f:
                benchmark_items = json.load(f)
        except FileNotFoundError:
            print(f"错误: 在 {args.benchmark_file} 未找到Benchmark文件。")
            return

        for i, item in enumerate(benchmark_items):
            if "id" not in item:
                item["id"] = f"{os.path.basename(item['images'][0])}_{i}" # 使用图片名和索引创建更唯一的ID
        
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
        
        print(f"\n推理完成！结果已保存到: {inference_output_path}")
        
    elif args.mode == 'evaluation':
        # --- 评估模式 ---
        print("--- 进入 [评估模式] ---")
        evaluation_input_path = args.evaluation_input_file if args.evaluation_input_file else Inference_output_file

        if not args.gemini_api_key:
            print("错误: 评估模式需要Gemini API密钥。请使用 --gemini_api_key 参数。")
            return
            
        try:
            with open(args.benchmark_file, 'r', encoding='utf-8') as f:
                benchmark_data_list = json.load(f)
                benchmark_data_map = {}
                for i, item in enumerate(benchmark_data_list):
                    # 使用与推理时相同的ID生成逻辑
                    item_id = item.get("id", f"{os.path.basename(item['images'][0])}_{i}")
                    benchmark_data_map[item_id] = item

            with open(evaluation_input_path, 'r', encoding='utf-8') as f:
                model_answers = [json.loads(line) for line in f]
        except FileNotFoundError as e:
            print(f"错误: 无法找到输入文件 - {e}")
            return
        
        items_to_evaluate = []
        for answer in model_answers:
            item_id = answer.get("id")
            if item_id in benchmark_data_map:
                full_item = benchmark_data_map[item_id]
                full_item['model_answer'] = answer['model_answer']
                items_to_evaluate.append(full_item)
            else:
                print(f"警告: 在原始benchmark数据中找不到ID为 '{item_id}' 的项，跳过。")
        
        if not items_to_evaluate:
            print("错误: 没有可供评估的数据。请检查ID是否匹配。")
            return
            
        semaphore = asyncio.Semaphore(args.concurrency)
        
        genai.configure(api_key=args.gemini_api_key)
        gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        
        evaluation_tasks = [run_evaluation(item, gemini_model, semaphore) for item in items_to_evaluate]
        final_results_list = await tqdm_asyncio.gather(*evaluation_tasks, desc="Evaluating")

        # 计算摘要信息
        summary = calculate_summary(
            results=final_results_list,
            model_name=Test_Model,
            benchmark_file=args.benchmark_file,
            evaluator_model=GEMINI_MODEL_NAME
        )

        # 构建最终的输出对象
        final_output_object = {
            "summary": summary,
            "results": final_results_list
        }

        # 保存最终的完整评估结果对象
        output_dir = os.path.dirname(args.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(final_output_object, f, indent=2, ensure_ascii=False)
        
        # 在终端打印漂亮的摘要信息
        print("\n--- 评估完成！摘要如下 ---")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print("--------------------------")
        print(f"\n完整结果已保存到: {args.output_file}")

    else:
        print("错误: 模式不明确。请使用 --mode 'inference' 或 'evaluation' 来指定运行模式。")

if __name__ == "__main__":
    asyncio.run(main())