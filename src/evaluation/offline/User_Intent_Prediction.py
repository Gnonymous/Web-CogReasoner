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
import numpy as np # 引入numpy用于更安全地计算平均值

# --- 配置项 ---
# UI-TARs CogReasoner Qwen2.5-VL-7B
Test_Model = "CogReasoner"
# 注意：使用f-string在这里定义全局变量可能不是最佳实践，但在脚本顶部可以接受
OUTPUT_JSON_PATH = f"/code/CogReasoner/Code/Evalaute/Result/Test-{Test_Model}-User_Intent_Prediction.json"
Inference_output_file = f"/code/CogReasoner/Code/Evalaute/Result/Raw_Answer-{Test_Model}-User_Intent_Prediction.jsonl"
VLLM_API_URL = "http://localhost:8080/v1/chat/completions"
GEMINI_MODEL_NAME = 'gemini-2.5-flash-lite-preview-06-17'
MAX_CONCURRENT_REQUESTS = 2 # 控制并发请求数，可根据您的硬件和API限制调整

# --- 提示模板 ---
def get_gemini_evaluator_prompt(ground_truth: str, model_answer: str) -> str:
    """为Gemini评估器创建详细的提示。"""
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

# --- 辅助函数 ---
def encode_image_to_base64(image_path: str) -> str:
    """将图片文件编码为base64字符串。"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"警告: 在路径 {image_path} 未找到图片文件")
        return None

# --- 【修改】此函数现在接受一个图片base64编码的列表 ---
def create_vllm_payload(user_prompt: str, image_base64_list: List[str]) -> Dict[str, Any]:
    """为vLLM的OpenAI兼容API创建JSON负载，支持多张图片。"""
    # 构建消息内容，文本在前，图片在后
    content = [{"type": "text", "text": user_prompt}]
    for image_base64 in image_base64_list:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_base64}"}
        })

    return {
        "model": "qwen2vl", # !!重要!! 确保这是您在vLLM中加载的模型名
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ],
        "max_tokens": 1024,
        "temperature": 0.1
    }

# --- 核心异步函数 ---
async def run_inference(item: Dict[str, Any], session: aiohttp.ClientSession, semaphore: asyncio.Semaphore) -> Dict[str, Any]:
    """仅执行推理阶段，并返回包含答案的关键信息。"""
    async with semaphore:
        # --- 【修改】处理多张图片路径 ---
        image_paths = item['images'] # 获取图片路径列表
        image_id_base = os.path.basename(image_paths[0]) if image_paths else "no_image"
        
        user_prompt = item['messages'][0]['content']
        model_answer = None

        # --- 【修改】编码所有图片 ---
        image_base64_list = [encode_image_to_base64(p) for p in image_paths]
        # 过滤掉编码失败的图片（比如文件不存在）
        image_base64_list = [b64 for b64 in image_base64_list if b64 is not None]

        if not image_base64_list:
            model_answer = "Error: All image files not found or failed to encode."
        else:
            # --- 【修改】使用新的payload创建函数 ---
            payload = create_vllm_payload(user_prompt, image_base64_list)
            try:
                async with session.post(VLLM_API_URL, json=payload, timeout=120) as response:
                    response.raise_for_status()
                    result = await response.json()
                    model_answer = result['choices'][0]['message']['content']
            except Exception as e:
                model_answer = f"Error during vLLM inference: {e}"
        # 只返回包含模型答案的关键信息，用于写入jsonl
        return {"id": item.get("id", image_id_base), "model_answer": model_answer}

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

# --- 【修改】此函数现在从评估结果中提取与Prompt一致的字段 ---
def calculate_summary(results: List[Dict[str, Any]], model_name: str, benchmark_file: str, evaluator_model: str) -> Dict[str, Any]:
    """计算评估结果的摘要统计信息。"""
    # 与 get_gemini_evaluator_prompt 中的JSON结构保持一致
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
        # 从评估JSON中提取分数
        scores["evidence_alignment"].append(eval_data.get("evidence_alignment_score", 0))
        scores["intent_accuracy"].append(eval_data.get("intent_accuracy_score", 0))
        scores["reasoning_quality"].append(eval_data.get("reasoning_quality_score", 0))
        scores["overall"].append(eval_data.get("overall_score", 0))

    # 使用numpy.mean来安全地处理空列表（如果所有评估都失败）
    average_scores = {
        "evidence_alignment_avg": round(np.mean(scores["evidence_alignment"]).item() if scores["evidence_alignment"] else 0, 3),
        "intent_accuracy_avg": round(np.mean(scores["intent_accuracy"]).item() if scores["intent_accuracy"] else 0, 3),
        "reasoning_quality_avg": round(np.mean(scores["reasoning_quality"]).item() if scores["reasoning_quality"] else 0, 3),
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
    parser.add_argument("--gemini_api_key", default=os.getenv("GEMINI_API_KEY", ""), help="您的Google AI Studio API密钥。") # 建议替换为实际密钥或环境变量
    parser.add_argument("--benchmark_file", default="/code/CogReasoner/Test/User_Intent_Prediction.json", help="包含测试数据的JSON文件路径。")
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

        # 为每个项目生成唯一的ID，以便在评估阶段进行匹配
        for i, item in enumerate(benchmark_items):
            if "id" not in item:
                # 使用第一个图片的文件名和索引创建唯一的ID
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
        
        print(f"\n推理完成！结果已保存到: {inference_output_path}")
        
    elif args.mode == 'evaluation':
        # --- 评估模式 ---
        print("--- 进入 [评估模式] ---")
        evaluation_input_path = args.evaluation_input_file if args.evaluation_input_file else Inference_output_file

        if not args.gemini_api_key or args.gemini_api_key == "YOUR_GEMINI_API_KEY":
            print("错误: 评估模式需要Gemini API密钥。请使用 --gemini_api_key 参数。")
            return
            
        try:
            with open(args.benchmark_file, 'r', encoding='utf-8') as f:
                benchmark_data_list = json.load(f)
                benchmark_data_map = {}
                # 使用与推理时相同的ID生成逻辑来构建映射
                for i, item in enumerate(benchmark_data_list):
                    img_path = item.get('images', [f'no_image_{i}'])[0]
                    item_id = item.get("id", f"{os.path.basename(img_path)}_{i}")
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
    # 注意：请将下面的 YOUR_GEMINI_API_KEY 替换为您的真实密钥，或通过命令行参数 --gemini_api_key 提供。
    # 为了安全，建议从环境变量或配置文件中读取密钥。
    # os.environ['GEMINI_API_KEY'] = 'YOUR_GEMINI_API_KEY'
    # parser.add_argument("--gemini_api_key", default=os.getenv("GEMINI_API_KEY"), ...)
    asyncio.run(main())