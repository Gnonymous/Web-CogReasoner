import os
import sys
import json
import base64
import re
import asyncio
import aiofiles
from tqdm.asyncio import tqdm_asyncio # Used for progress bar in async tasks
import boto3 # Import boto3 for AWS Bedrock interaction
from rouge import Rouge

Test_Model = "Claude" # Define the model name for testing

# ===== Configuration Items =====
TEST_JSON_PATH = "/code/CogReasoner/Test/VisualWebBench_webqa.json"  # 测试数据路径
MODEL_NAME = "us.anthropic.claude-sonnet-4-20250514-v1:0" # Specify the Claude model name for inference
MAX_SAMPLE = 245  # Maximum number of samples to test
MAX_CONCURRENT_REQUESTS = 1 # Maximum concurrent requests (set to 10 for async processing)
ACCURACY_PRINT_INTERVAL = 10 # Print current accuracy after processing this many samples
OUTPUT_JSON_PATH = f"/code/CogReasoner/Code/Evalaute/Result/Test-{Test_Model}-VisualWebBench_WebQA.json" # Path where inference results will be saved
FIXED_PROMPT = (
    "You are given a screenshot of a webpage. Please generate the main text within the screenshot, "
    "which can be regarded as the heading of the webpage.\n\n"
    "You should directly tell me the main content, and do not output any explanation or any other contents."
)
# ===== AWS Bedrock Claude Client Class =====
class BedrockClaudeClient:
    """
    A client for interacting with the Claude model on AWS Bedrock.
    AWS credentials are configured directly in this code (for demonstration).
    """
    def __init__(self, access_key, secret_key, region_name, model_id):
        """
        Initializes the Bedrock runtime client with provided keys and region info.
        """
        self.model_id = model_id
        
        try:
            self.bedrock_client = boto3.client(
                service_name='bedrock-runtime',
                region_name=region_name,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key
            )
            print(f"Boto3 client successfully created in region '{region_name}' for model '{self.model_id}'!")
        except Exception as e:
            raise ConnectionError(f"Failed to create Bedrock client: {e}. Please check your AWS credentials and region name.")

    def _parse_data_url(self, data_url):
        """
        Parses a Data URL (e.g., data:image/png;base64,iVBOR...)
        Extracts media_type and Base64 data.
        """
        if not data_url.startswith("data:"):
            print(f"Warning: Not a standard Data URL format: {data_url}")
            return None, None

        parts = data_url.split(',', 1)
        if len(parts) < 2:
            print(f"Warning: Incomplete Data URL format: {data_url}")
            return None, None

        metadata = parts[0][len("data:"):].split(';')
        media_type = metadata[0]
        base64_data = parts[1]

        if "base64" not in metadata:
            print(f"Warning: Data URL does not contain 'base64' encoding identifier: {data_url}")
            return None, None

        return base64_data, media_type

    # This method is kept as `def` (synchronous) because `boto3` client calls are synchronous.
    # It will be called within `asyncio.to_thread` in `process_item` to avoid blocking the event loop.
    def chat(self, messages, max_tokens=1024, temperature=0.7):
        """
        Sends messages to the Claude model and gets a reply.
        This function is now fully compatible with OpenAI-format message lists,
        including handling system messages and embedded base64 image_url.
        """
        if not hasattr(self, 'bedrock_client'):
            raise RuntimeError("Bedrock client not successfully initialized.")

        claude_system_message = None
        claude_messages_payload = []

        # Convert OpenAI format to Claude Bedrock format
        for openai_msg in messages:
            role = openai_msg.get("role")
            content = openai_msg.get("content")

            if role == "system":
                claude_system_message = content
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
                                        "source": {
                                            "type": "base64",
                                            "media_type": media_type,
                                            "data": base64_data
                                        }
                                    })
                            else:
                                print(f"Warning: Could not parse image data from Data URL {url}, skipping this content block.")
                        else:
                            print(f"Warning: Unsupported OpenAI content type: {item.get('type')}. Skipping this content block.")
                
                if claude_content_blocks:
                    claude_messages_payload.append({"role": role, "content": claude_content_blocks})
                else:
                    print(f"Warning: '{role}' role message has no valid content, skipping.")

            else:
                print(f"Warning: Unsupported OpenAI message role: {role}. Skipping this message.")

        if not claude_messages_payload:
            raise ValueError("No valid 'user' or 'assistant' role messages to send to Claude after conversion.")

        # Build the request body
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": claude_messages_payload
        }
        # Add system message if it exists
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

            usage = response_body.get('usage', {})
            prompt_tokens = usage.get('input_tokens', 0)
            completion_tokens = usage.get('output_tokens', 0)

            return {
                "response_text": response_text,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens
            }
        except Exception as e:
            error_message = str(e)
            if hasattr(e, 'response') and 'Error' in e.response:
                error_message = f"{e.response['Error'].get('Code', '')}: {e.response['Error'].get('Message', '')}"
            raise RuntimeError(f"Error calling Claude model: {error_message}")


def eval_webqa(preds, golds, **kwargs):
    """
    计算 WebQA 的 F1 分数。
    preds: 预测答案的列表。
    golds: 参考答案的列表的列表 (每个问题可以有多个参考答案)。
    """
    assert len(preds) == len(golds), "预测数量和参考答案数量必须一致"
    f1_scores = []
    # 注意：Rouge() 实例在循环外创建以提高效率
    rouge = Rouge(metrics=['rouge-1'])
    for pred, gold_list in zip(preds, golds):
        if not pred:
            pred = " "  # 避免空字符串导致ROUGE计算异常
        
        # 计算当前预测与所有参考答案的 F1 分数，并取最大值
        # gold_list 是当前问题的正确答案列表，例如 ['Sawfish'] 或 ['Answer A', 'Answer B']
        try:
            current_f1 = max([rouge.get_scores([pred], [gold], avg=True)['rouge-1']['f'] for gold in gold_list])
            f1_scores.append(current_f1)
        except Exception as e:
            # 如果发生错误（例如 gold_list 为空），则记录为0分并打印警告
            print(f"Warning: Could not compute F1 score for pred='{pred}' and gold_list='{gold_list}'. Error: {e}")
            f1_scores.append(0.0)

    # 确保 f1_scores 不为空，以避免除以零的错误
    if not f1_scores:
        return dict(f1=0.0)

    return dict(
        f1=sum(f1_scores) / len(f1_scores) * 100
    )



# ===== Asynchronously Process Single Sample =====
async def process_item(index, item, sem, claude_client_instance, stats):
    async with sem:
        image_path = item["images"][0]
        
        # --- 关键修改 ---
        # `eval_webqa` 需要一个答案列表，所以我们将单个答案包装成列表
        # 即使只有一个正确答案，也需要是列表形式，例如 ['Sawfish']
        ground_truth = [item["messages"][1]["content"].strip()]
        
        user_prompt = item["messages"][0]["content"] # user_prompt 是包含 <image> 和问题的完整内容

        # 读取并编码图片
        async with aiofiles.open(image_path, "rb") as f:
            content = await f.read()
        encoded_image = base64.b64encode(content).decode("utf-8")
        image_data_uri = f"data:image/png;base64,{encoded_image}"
        prompt_text = user_prompt.replace("<image>\n", "").strip()
        # 构造消息
        messages =[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_data_uri}},
                            {"type": "text", "text": prompt_text},
                        ],
                    },
                ]
        try:
            
            # Send inference request to the model
            # The ClaudeClient.chat method is synchronous, so we run it in a thread pool
            response_data = await asyncio.to_thread(
                claude_client_instance.chat,
                messages = messages,
                max_tokens=2048, # Limit max length of generated text
                temperature=0.1 # Control randomness of generated text
            )
            pred_text = response_data['response_text'].strip() # Extract model generated text content
        except Exception as e:
            pred_text = f"[ERROR] {str(e)}" # Capture exception and log error message
        await asyncio.sleep(10) # 暂停0.5秒。根据你的RPS限制调整这个值。
                                 # 例如，如果RPS是2，你可能需要等待0.5秒。
        return {
            "image": image_path,
            "ground_truth": ground_truth, # ground_truth 现在是一个列表
            "prediction": pred_text,
        }

# ===== Main Function =====
async def main():
    """
    Main execution function, responsible for loading test data, creating and running
    asynchronous tasks, collecting results, and saving them.
    """
    # AWS credentials and model ID (fill in your actual values)
    # WARNING: Hardcoding credentials directly is insecure. For production, use environment variables,
    # AWS CLI configuration, or IAM roles.
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID", "") # REPLACE WITH YOUR AWS ACCESS KEY ID
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY", "") # REPLACE WITH YOUR AWS SECRET ACCESS KEY
    aws_region_name = "us-east-1"
    aws_model_id = MODEL_NAME # Using MODEL_NAME from config

    # Initialize AWS Bedrock Claude client
    try:
        claude_client = BedrockClaudeClient(
            access_key=aws_access_key_id,
            secret_key=aws_secret_access_key,
            region_name=aws_region_name,
            model_id=aws_model_id
        )
    except Exception as e:
        print(f"Failed to initialize Bedrock Claude client: {e}")
        sys.exit(1) # Exit program

    # Read test set JSON file
    with open(TEST_JSON_PATH, "r", encoding="utf-8") as f:
        test_data = json.load(f)[:MAX_SAMPLE] # Load data and truncate based on MAX_SAMPLE

    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    stats = {"total": 0, "correct": 0} # Initialize statistics

    tasks = [process_item(i, item, sem, claude_client, stats) for i, item in enumerate(test_data)]

    print(f"\n🚀 Starting evaluation for WebQA on {len(tasks)} samples...\n")
    results = await tqdm_asyncio.gather(*tasks)

    predictions = [r["prediction"] for r in results]
    references = [r["ground_truth"] for r in results] # 这现在是一个列表的列表

    # --- 关键修改 ---
    # 调用新的评估函数
    metrics = eval_webqa(predictions, references)

    output = {
        "task": "WebQA",
        "model": Test_Model,
        "metrics": metrics,
        "results": results,
    }

    # 保存结果
    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Evaluation Complete!")
    print(f"📊 Metrics: {json.dumps(metrics, indent=2)}")
    print(f"📁 Results saved at: {OUTPUT_JSON_PATH}")


# ===== Entry Point =====
if __name__ == "__main__":
    asyncio.run(main()) # Run the main asynchronous function
    sys.exit(0) # Force exit to prevent the async event loop from hanging