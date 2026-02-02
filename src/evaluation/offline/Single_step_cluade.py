import os
import sys
import json
import base64
import re
import asyncio
import aiofiles
from tqdm.asyncio import tqdm_asyncio
import boto3  # <--- 新增: 从脚本 B 引入

# ===================================================================
# ===== 1. 合并后的配置项 =====
# ===================================================================

# --- 任务配置 (来自脚本 A) ---
TEST_JSON_PATH = "/code/CogReasoner/Test/MultiStep_Selected_OnePerSite_step01_FinalAction.json"
MAX_SAMPLE = 70
MAX_CONCURRENT_REQUESTS = 1  # Claude的RPS较低，如果遇到限流错误，请调低此值
ACCURACY_PRINT_INTERVAL = 10

# --- 模型和输出配置 (合并和修改) ---
Model_name = "Claude"  # 用于输出文件的模型名称
BEDROCK_MODEL_ID = "us.anthropic.claude-sonnet-4-20250514-v1:0" # <--- 修改: 指定你的Claude模型ID
OUTPUT_JSON_PATH = f"/code/CogReasoner/Code/Evalaute/Result/Test-{Model_name}-Single_Step.json"

# --- AWS Bedrock 凭证 (来自脚本 B) ---
# 警告: 直接在代码中硬编码凭证是不安全的做法。
# 推荐使用环境变量、AWS配置文件或IAM角色。
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")  # <--- 修改: 填入你的AWS Access Key
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "") # <--- 修改: 填入你的AWS Secret Key
AWS_REGION_NAME = "us-east-1"  # <--- 修改: 填入你的AWS区域

# ===================================================================
# ===== 2. AWS Bedrock Claude 客户端 (从脚本 B 完整引入) =====
# ===================================================================

class BedrockClaudeClient:
    """
    用于与 AWS Bedrock 上的 Claude 模型交互的客户端。
    """
    def __init__(self, access_key, secret_key, region_name, model_id):
        """
        初始化 Bedrock 运行时客户端。
        """
        self.model_id = model_id
        try:
            self.bedrock_client = boto3.client(
                service_name='bedrock-runtime',
                region_name=region_name,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key
            )
            print(f"Boto3 客户端成功创建，区域: '{region_name}', 模型: '{self.model_id}'!")
        except Exception as e:
            raise ConnectionError(f"创建 Bedrock 客户端失败: {e}。请检查您的 AWS 凭证和区域名称。")

    def _parse_data_url(self, data_url):
        if not data_url.startswith("data:"): return None, None
        parts = data_url.split(',', 1)
        if len(parts) < 2: return None, None
        metadata = parts[0][len("data:"):].split(';')
        media_type = metadata[0]
        base64_data = parts[1]
        if "base64" not in metadata: return None, None
        return base64_data, media_type

    # 注意: 此方法是同步的，因为它使用了同步的boto3库。
    # 我们将在异步代码中通过 asyncio.to_thread 来调用它。
    def chat(self, messages, max_tokens=2048, temperature=0.1):
        """
        将消息发送到 Claude 模型并获取回复。
        此函数兼容 OpenAI 格式的消息列表。
        """
        if not hasattr(self, 'bedrock_client'):
            raise RuntimeError("Bedrock 客户端未成功初始化。")

        claude_system_message = None
        claude_messages_payload = []

        # 将 OpenAI 格式转换为 Claude Bedrock 格式
        for openai_msg in messages:
            role = openai_msg.get("role")
            content = openai_msg.get("content")

            if role == "system":
                # Claude 通过顶层参数处理 system prompt
                if isinstance(content, str):
                    claude_system_message = content
                else:
                     print(f"警告: 不支持的 system content 类型: {type(content)}")
            elif role in ["user", "assistant"]:
                claude_content_blocks = []
                if isinstance(content, str):
                    claude_content_blocks.append({"type": "text", "text": content})
                elif isinstance(content, list):
                    for item in content:
                        if item.get("type") == "text":
                            claude_content_blocks.append({"type": "text", "text": item.get("text", "")})
                        elif item.get("type") == "image_url":
                            image_url_dict = item.get("image_url", {})
                            url = image_url_dict.get("url")
                            if url:
                                base64_data, media_type = self._parse_data_url(url)
                                if base64_data and media_type:
                                    claude_content_blocks.append({
                                        "type": "image",
                                        "source": {"type": "base64", "media_type": media_type, "data": base64_data}
                                    })
                if claude_content_blocks:
                    claude_messages_payload.append({"role": role, "content": claude_content_blocks})

        if not claude_messages_payload:
            raise ValueError("转换后没有有效的 'user' 或 'assistant' 消息可以发送给 Claude。")

        # 构建请求体
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": claude_messages_payload
        }
        if claude_system_message:
            body["system"] = claude_system_message

        try:
            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body)
            )
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
            raise RuntimeError(f"调用 Claude 模型出错: {error_message}")


# ===================================================================
# ===== 3. 核心函数 (来自脚本 A, 几乎无变化) =====
# ===================================================================

def extract_action(text: str):
    """从模型输出中提取标准化的动作字符串 (无变化)"""
    if not text:
        return None
    type_match = re.search(r"Action:\s+type\s+\[(\d+)\]\s+\((.*?)\)", text, re.IGNORECASE)
    if type_match:
        node_id = type_match.group(1)
        value = type_match.group(2)
        return f"TYPE({node_id}, {value})"
    simple_match = re.search(r"Action:\s+(\w+)\s+\[(\d+)\]", text, re.IGNORECASE)
    if simple_match:
        action = simple_match.group(1).upper()
        node_id = simple_match.group(2)
        return f"{action}({node_id})"
    return None

def parse_ground_truth(gt_content: str):
    """解析可能包含多个答案的 Ground Truth 字符串 (无变化)"""
    action_parts = gt_content.split(';')
    parsed_actions = [extract_action(part.strip()) for part in action_parts]
    return [action for action in parsed_actions if action is not None]

def compare_actions(prediction: str, ground_truth_list: list) -> bool:
    """智能比较函数，特别处理 TYPE 动作 (无变化)"""
    if not prediction:
        return False

    pred_match = re.match(r"(\w+)\((\d+)", prediction)
    if not pred_match:
        return False
    pred_action_type = pred_match.group(1)
    pred_node_id = pred_match.group(2)

    for gt_action in ground_truth_list:
        if prediction == gt_action:
            return True
        if pred_action_type == "TYPE" and gt_action.startswith("TYPE("):
            gt_match = re.match(r"TYPE\((\d+)", gt_action)
            if gt_match and pred_node_id == gt_match.group(1):
                return True
    return False

# ===================================================================
# ===== 4. 异步处理函数 (关键修改) =====
# ===================================================================

async def process_item(index, item, sem, claude_client, stats):
    """
    修改此函数以使用 claude_client 而不是 OpenAI client。
    """
    async with sem:
        image_paths = item["images"]
        prompt = item["messages"][0]["content"]
        gt_json_str = item["messages"][-1]["content"]
        gt_answers_list = parse_ground_truth(gt_json_str)

        if not gt_answers_list:
            # ... (错误处理部分无变化)
            return {"match": False, "raw_model_output": "Ground truth format is invalid."}

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
                # ... (错误处理部分无变化)
                return {"match": False, "raw_model_output": f"Image not found: {path}"}

        # prompt 格式保持脚本 A 的逻辑
        messages = [
            {
                "role": "user",
                "content": image_contents + [
                    {
                        "type": "text",
                        # <--- 修改: Prompt 可以根据Claude的特性微调，但格式保持不变
                        "text": "Based on the provided image, task description, and AxTree, please output the element IDs required to complete the task. The expected answer format is: `Action: click [AxTree ID]`. " + prompt.strip(),
                    }
                ],
            },
        ]

        # <--- 关键修改: 模型调用部分 ---
        pred_text = ""
        try:
            # 使用 asyncio.to_thread 异步调用同步的 boto3 客户端
            response_data = await asyncio.to_thread(
                claude_client.chat,
                messages=messages,
                temperature=0.1,
                max_tokens=2048
            )
            pred_text = response_data['response_text'].strip()
        except Exception as e:
            pred_text = f"[ERROR] {str(e)}"
        
        # 答案提取和比较逻辑保持不变
        pred_answer = extract_action(pred_text)
        match = compare_actions(pred_answer, gt_answers_list)
        
        # 统计数据更新逻辑保持不变
        stats["total"] += 1
        stats["correct"] += int(match)
        if stats["total"] > 0 and stats["total"] % ACCURACY_PRINT_INTERVAL == 0:
            acc = stats["correct"] / stats["total"] * 100
            print(f"\n📊 Step {stats['total']}: Accuracy = {acc:.2f}%\n")

        # 返回结果字典保持不变
        return {
            "images": image_paths,
            "prompt": prompt,
            "ground_truth": ";".join(gt_answers_list),
            "prediction": pred_answer,
            "match": match,
            "raw_model_output": pred_text
        }

# ===================================================================
# ===== 5. 主函数 (修改以初始化和传递新客户端) =====
# ===================================================================

async def main():
    # <--- 新增: 初始化 Bedrock Claude 客户端 ---
    try:
        claude_client = BedrockClaudeClient(
            access_key=AWS_ACCESS_KEY_ID,
            secret_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION_NAME,
            model_id=BEDROCK_MODEL_ID
        )
    except Exception as e:
        print(f"初始化 Bedrock 客户端失败: {e}")
        sys.exit(1)

    with open(TEST_JSON_PATH, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    if MAX_SAMPLE is not None and MAX_SAMPLE > 0:
        test_data = test_data[:MAX_SAMPLE]

    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    stats = {"total": 0, "correct": 0}

    # <--- 修改: 将 claude_client 实例传递给 process_item ---
    tasks = [process_item(i, item, sem, claude_client, stats) for i, item in enumerate(test_data)]

    print(f"\n🚀 开始评估 {len(tasks)} 个样本，使用模型 '{BEDROCK_MODEL_ID}'...\n")
    results = await tqdm_asyncio.gather(*tasks)

    # 后续的结果处理、统计和保存逻辑完全保持不变
    valid_results = [r for r in results if r is not None]
    if not valid_results:
        print("\n❌ 没有处理任何有效样本。评估无法完成。")
        return

    final_total = stats["total"]
    final_correct = stats["correct"]
    accuracy = (final_correct / final_total * 100) if final_total > 0 else 0
    errors = [r for r in valid_results if not r["match"]]

    output = {
        "model_name": Model_name,
        "metrics": {
            "total_processed": final_total,
            "correct": final_correct,
            "accuracy": accuracy
        },
        "results": valid_results,
        "errors": errors
    }

    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 评估完成")
    print(f"🎯 准确率: {accuracy:.2f}% ({final_correct}/{final_total})")
    print(f"📁 结果已保存至: {OUTPUT_JSON_PATH}")

    if errors:
        print("\n❌ 错误样本示例 (最多 5 个):")
        for r in errors[:5]:
            print(f"- Images       : {', '.join(r['images'])}")
            print(f"  Ground Truth : {r['ground_truth']}")
            print(f"  Prediction   : {r['prediction']}")
            raw_output_snippet = r['raw_model_output'].replace('\n', ' ')
            if len(raw_output_snippet) > 200:
                raw_output_snippet = "..." + raw_output_snippet[-200:]
            print(f"  Raw Output   : {raw_output_snippet}\n")
    # 注意: 不再需要 client.aclose()

# ===================================================================
# ===== 6. 启动入口 (无变化) =====
# ===================================================================

if __name__ == "__main__":
    # 在运行前，请确保已填写 AWS 凭证
    if "YOUR_AWS_ACCESS_KEY_ID" in AWS_ACCESS_KEY_ID:
        print("错误: 请在脚本中填写您的 AWS_ACCESS_KEY_ID 和 AWS_SECRET_ACCESS_KEY。")
        sys.exit(1)
    asyncio.run(main())