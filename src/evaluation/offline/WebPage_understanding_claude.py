import asyncio
import aiohttp
import google.generativeai as genai
import json
import os
import argparse
import io
import base64
from tqdm.asyncio import tqdm_asyncio
from typing import Dict, Any, List
from PIL import Image
from datetime import datetime
import numpy as np
import boto3 # 引入boto3用于与AWS Bedrock交互

# --- 配置项 ---
# 1. 要测试的模型
Test_Model = "Claude" 

# 2. 输出文件路径
OUTPUT_JSON_PATH = f"/code/CogReasoner/Code/Evalaute/Result/Test-{Test_Model}-WebPage_Understanding_77.json"
Inference_output_file = f"/code/CogReasoner/Code/Evalaute/Result/Raw_Answer-{Test_Model}-WebPage_Understanding_77.jsonl"

# 3. 评测模型 (Gemini)
GEMINI_MODEL_NAME = 'gemini-2.5-flash-lite-preview-06-17'
MAX_CONCURRENT_REQUESTS = 1 # 控制并发请求数

# 4. 推理模型 (Claude on AWS Bedrock) - 在此处硬编码您的凭证和配置
# !!重要!! 请在此处填入您的真实凭证
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "") # <--- 在这里填入您的 AWS Access Key ID
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "") # <--- 在这里填入您的 AWS Secret Access Key
AWS_REGION_NAME = "us-east-1"  # 例如 "us-east-1"
CLAUDE_MODEL_ID = "us.anthropic.claude-sonnet-4-20250514-v1:0" # 您希望使用的Claude模型ID

# --- AWS Bedrock Claude 客户端 (来自您的示例) ---
class BedrockClaudeClient:
    """
    一个用于与 AWS Bedrock 上的 Claude 模型交互的客户端。
    """
    def __init__(self, access_key: str, secret_key: str, region_name: str, model_id: str):
        self.model_id = model_id
        try:
            self.bedrock_client = boto3.client(
                service_name='bedrock-runtime',
                region_name=region_name,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key
            )
            print(f"Boto3 客户端成功创建，区域：'{region_name}', 模型：'{self.model_id}'!")
        except Exception as e:
            raise ConnectionError(f"创建 Bedrock 客户端失败: {e}。请检查您的 AWS 凭证和区域名称。")

    def _parse_data_url(self, data_url: str):
        try:
            header, encoded = data_url.split(",", 1)
            media_type = header.split(";")[0].split(":")[1]
            return encoded, media_type
        except Exception:
            print(f"警告: 无法解析 Data URL: {data_url[:30]}...")
            return None, None

    def chat(self, messages: List[Dict[str, Any]], max_tokens: int = 1024, temperature: float = 0.1) -> Dict[str, Any]:
        if not hasattr(self, 'bedrock_client'):
            raise RuntimeError("Bedrock 客户端未成功初始化。")

        claude_system_message = None
        claude_messages_payload = []

        for openai_msg in messages:
            role = openai_msg.get("role")
            content = openai_msg.get("content")
            if role == "system":
                claude_system_message = content
            elif role == "user":
                claude_content_blocks = []
                if isinstance(content, list):
                    for item in content:
                        if item.get("type") == "text":
                            claude_content_blocks.append({"type": "text", "text": item.get("text", "")})
                        elif item.get("type") == "image_url":
                            url = item.get("image_url", {}).get("url")
                            if url:
                                base64_data, media_type = self._parse_data_url(url)
                                if base64_data and media_type:
                                    claude_content_blocks.append({
                                        "type": "image",
                                        "source": {"type": "base64", "media_type": media_type, "data": base64_data}
                                    })
                claude_messages_payload.append({"role": "user", "content": claude_content_blocks})

        if not claude_messages_payload:
            raise ValueError("转换后没有有效的 'user' 角色消息可以发送给 Claude。")

        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": claude_messages_payload
        }
        if claude_system_message:
            body["system"] = claude_system_message

        try:
            response = self.bedrock_client.invoke_model(modelId=self.model_id, body=json.dumps(body))
            response_body = json.loads(response.get('body').read())
            response_text = ""
            if response_body.get('content'):
                for content_block in response_body['content']:
                    if content_block.get('type') == 'text':
                        response_text += content_block['text']
            return {"response_text": response_text}
        except Exception as e:
            error_message = str(e)
            if hasattr(e, 'response') and 'Error' in e.response:
                error_message = f"{e.response['Error'].get('Code', '')}: {e.response['Error'].get('Message', '')}"
            raise RuntimeError(f"调用 Claude 模型时出错: {error_message}")


def get_gemini_evaluator_prompt(ground_truth: str, model_answer: str) -> str:
    """
    为网页综合分析任务创建一个详细的评估Prompt。
    该Prompt旨在评估模型对网页的结构理解、关键元素分析和总结能力。
    """
    return f"""You are a meticulous and impartial AI evaluator for a web UI understanding benchmark. Your task is to assess the quality of a candidate model's comprehensive webpage analysis by comparing it strictly against a ground truth reference.

Your evaluation must be based *exclusively* on the information provided in the "Ground Truth Answer". Do not use any external knowledge or make assumptions beyond what is written in the ground truth.

Evaluate the candidate answer on three specific aspects:
1.  **Structure and Layout Analysis**: How well does the model describe the overall structure of the webpage (e.g., navigation bar, main content area, sidebars, footer)? Compare this to the "Webpage Layout Description" in the ground truth.
2.  **Key Element Analysis**: How well does the model identify and analyze the key interactive elements? Assess if the chosen elements are relevant and if their description, function, and predicted user interaction match the details in the ground truth's "Key Element Analysis" section.
3.  **Summary and Coherence**: How well does the model summarize its findings? Is the summary accurate, concise, and logically consistent with the preceding analysis, as reflected in the ground truth's "Summary" section?

**[Ground Truth Answer]**
{ground_truth}
---
**[Candidate Model's Answer]**
{model_answer}
---

**Evaluation Criteria & Scoring:**
- **Score 1:** Completely incorrect or missing. The analysis is irrelevant or fails to address the core aspects of the ground truth.
- **Score 2:** Mostly incorrect. The analysis identifies some elements but describes them inaccurately or misses the main points of the ground truth.
- **Score 3:** Partially correct. The analysis captures some key aspects of the structure and elements but misses significant details or contains notable inaccuracies when compared to the ground truth.
- **Score 4:** Mostly correct. The analysis is largely accurate and comprehensive, with only minor inaccuracies or omissions compared to the ground truth.
- **Score 5:** Fully and accurately captures all relevant information and insights present in the ground truth, demonstrating a complete and nuanced understanding.

Your response MUST be a single, valid JSON object, adhering to the following structure. Do not add any text before or after the JSON object.

{{
  "structure_score": <integer_score from 1-5>,
  "structure_justification": "<Your brief justification for the structure score, referencing the 'Webpage Layout Description' in the ground truth>",
  "element_analysis_score": <integer_score from 1-5>,
  "element_analysis_justification": "<Your brief justification for the element analysis score, referencing the 'Key Element Analysis' details (e.g., element choice, function, interaction prediction) in the ground truth>",
  "summary_score": <integer_score from 1-5>,
  "summary_justification": "<Your brief justification for the summary score, referencing the 'Summary' section in the ground truth>",
  "overall_score": <A final holistic integer score from 1 to 5, considering all aspects>,
  "overall_justification": "<A final summary of the model's overall performance, highlighting its main strengths and weaknesses on this example>"
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


def create_model_payload(user_prompt: str, image_base64: str) -> Dict[str, Any]:
    """为模型创建兼容OpenAI格式的JSON负载。"""
    return {
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
async def run_inference(item: Dict[str, Any], claude_client: BedrockClaudeClient, semaphore: asyncio.Semaphore) -> Dict[str, Any]:
    """执行推理阶段（使用Claude），并返回包含答案的关键信息。"""
    async with semaphore:
        image_path = item['images'][0]
        MAX_DIMENSION = 8000 
        user_prompt = item['messages'][0]['content']
        # model_answer = None
        # image_base64 = encode_image_to_base64(image_path)
        image_base64=""
        with Image.open(image_path) as img:
    # 检查图片的宽和高
            width, height = img.size
            if width > MAX_DIMENSION or height > MAX_DIMENSION:
        # 如果超过限制，计算缩放比例
                ratio = MAX_DIMENSION / max(width, height)
                new_width = int(width * ratio)
                new_height = int(height * ratio)
        
        # 使用高质量的LANCZOS算法进行缩放
                img = img.resize((new_width, new_height), Image.LANCZOS)

    # 将处理后的图片保存到内存缓冲区
            buffered = io.BytesIO()
    # 保持原格式（如果是PNG），否则用JPEG
            image_format = 'PNG' if img.format == 'PNG' else 'JPEG'
            img.save(buffered, format=image_format)
    
    # 从缓冲区获取字节数据并进行Base64编码
            image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        if not image_base64:
            model_answer = "Error: Image file not found."
        else:
            payload = create_model_payload(user_prompt, image_base64)
            try:
                response_data = await asyncio.to_thread(
                    claude_client.chat,
                    messages=payload['messages'],
                    max_tokens=payload['max_tokens'],
                    temperature=payload['temperature']
                )
                model_answer = response_data['response_text']
                await asyncio.sleep(10)
            except Exception as e:
                model_answer = f"Error during Claude inference: {e}"
        
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

def calculate_summary(results: List[Dict[str, Any]], model_name: str, benchmark_file: str, evaluator_model: str) -> Dict[str, Any]:
    """计算评估结果的摘要统计信息。此函数已更新以匹配新的评估维度。"""
    # 修改评分键以匹配新的Prompt
    scores = {
        "structure": [],
        "element_analysis": [],
        "summary": [],
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
        # 修改提取逻辑以匹配新的JSON键
        scores["structure"].append(eval_data.get("structure_score", 0))
        scores["element_analysis"].append(eval_data.get("element_analysis_score", 0))
        scores["summary"].append(eval_data.get("summary_score", 0))
        scores["overall"].append(eval_data.get("overall_score", 0))

    # 使用numpy.mean来安全地处理空列表（如果所有评估都失败）
    # 修改平均分计算的键
    average_scores = {
        "structure_avg": round(np.mean(scores["structure"]).item() if scores["structure"] else 0, 3),
        "element_analysis_avg": round(np.mean(scores["element_analysis"]).item() if scores["element_analysis"] else 0, 3),
        "summary_avg": round(np.mean(scores["summary"]).item() if scores["summary"] else 0, 3),
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

# --- 主程序入口 (已恢复硬编码) ---
async def main():
    parser = argparse.ArgumentParser(description="分阶段Benchmark工具：可独立进行推理或评估。")
    parser.add_argument("--gemini_api_key", default=os.getenv("GEMINI_API_KEY", ""), help="您的Google AI Studio API密钥。")
    parser.add_argument("--benchmark_file", default="/code/CogReasoner/Test/WebPage_Understanding_77.json", help="包含测试数据的JSON文件路径。")
    parser.add_argument("--output_file", default=OUTPUT_JSON_PATH, help="保存最终评估结果的JSON文件路径。")
    parser.add_argument("--concurrency", type=int, default=MAX_CONCURRENT_REQUESTS, help="最大并发请求数。")
    parser.add_argument("--inference_output_file", type=str, help="[推理模式] 推理结果要保存到的.jsonl文件路径。如果未提供，将使用默认路径。")
    parser.add_argument("--evaluation_input_file", default=Inference_output_file,type=str, help="[评估模式] 包含模型答案的.jsonl文件路径。")
    parser.add_argument("--mode", choices=['inference', 'evaluation'], help="明确选择脚本运行模式：'inference' 或 'evaluation'。")
    args = parser.parse_args()

    if not args.mode:
        args.mode = 'evaluation' if args.evaluation_input_file else 'inference'

    if args.mode == 'inference':
        print("--- 进入 [推理模式] (模型: Claude) ---")
        
        # 检查硬编码的凭证是否已填写
        if "YOUR_AWS" in AWS_ACCESS_KEY_ID or "YOUR_AWS" in AWS_SECRET_ACCESS_KEY:
            print("错误: 推理模式需要 AWS 凭证。请在脚本顶部的配置项中填入您的真实凭证。")
            return

        try:
            # 使用脚本顶部的全局变量初始化客户端
            claude_client = BedrockClaudeClient(
                access_key=AWS_ACCESS_KEY_ID,
                secret_key=AWS_SECRET_ACCESS_KEY,
                region_name=AWS_REGION_NAME,
                model_id=CLAUDE_MODEL_ID
            )
        except Exception as e:
            print(f"初始化 Claude 客户端失败: {e}")
            return

        inference_output_path = args.inference_output_file if args.inference_output_file else Inference_output_file
        
        try:
            with open(args.benchmark_file, 'r', encoding='utf-8') as f:
                benchmark_items = json.load(f)
        except FileNotFoundError:
            print(f"错误: 在 {args.benchmark_file} 未找到Benchmark文件。")
            return

        for i, item in enumerate(benchmark_items):
            if "id" not in item:
                item["id"] = f"{os.path.basename(item['images'][0])}_{i}"
        
        semaphore = asyncio.Semaphore(args.concurrency)
        
        inference_tasks = [run_inference(item, claude_client, semaphore) for item in benchmark_items]
        inference_results = await tqdm_asyncio.gather(*inference_tasks, desc="Inferring with Claude")

        output_dir = os.path.dirname(inference_output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with open(inference_output_path, 'w', encoding='utf-8') as f:
            for result in inference_results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        print(f"\n推理完成！结果已保存到: {inference_output_path}")
        
    elif args.mode == 'evaluation':
        print("--- 进入 [评估模式] (评估模型: Gemini) ---")
        evaluation_input_path = args.evaluation_input_file if args.evaluation_input_file else Inference_output_file
        if not args.gemini_api_key or "YOUR_GEMINI" in args.gemini_api_key:
            print("错误: 评估模式需要Gemini API密钥。请使用 --gemini_api_key 参数。")
            return
        try:
            with open(args.benchmark_file, 'r', encoding='utf-8') as f:
                benchmark_data_list = json.load(f)
                benchmark_data_map = {}
                for i, item in enumerate(benchmark_data_list):
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
        final_results_list = await tqdm_asyncio.gather(*evaluation_tasks, desc="Evaluating with Gemini")

        summary = calculate_summary(
            results=final_results_list,
            model_name=Test_Model,
            benchmark_file=args.benchmark_file,
            evaluator_model=GEMINI_MODEL_NAME
        )
        final_output_object = {"summary": summary, "results": final_results_list}

        output_dir = os.path.dirname(args.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(final_output_object, f, indent=2, ensure_ascii=False)
        
        print("\n--- 评估完成！摘要如下 ---")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print("--------------------------")
        print(f"\n完整结果已保存到: {args.output_file}")
    else:
        print("错误: 模式不明确。请使用 --mode 'inference' 或 'evaluation' 来指定运行模式。")

if __name__ == "__main__":
    asyncio.run(main())