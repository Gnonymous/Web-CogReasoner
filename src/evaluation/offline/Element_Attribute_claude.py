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
TEST_JSON_PATH = "/code/CogReasoner/Test/Element_Attribute_249.json" # Path to the test set JSON file
MODEL_NAME = "us.anthropic.claude-sonnet-4-20250514-v1:0" # Specify the Claude model name for inference
MAX_SAMPLE = 249 # Maximum number of samples to test
MAX_CONCURRENT_REQUESTS = 1 # Maximum concurrent requests (set to 10 for async processing)
ACCURACY_PRINT_INTERVAL = 10 # Print current accuracy after processing this many samples
OUTPUT_JSON_PATH = f"/code/CogReasoner/Code/Evalaute/Result/Test-{Test_Model}-Element_Attribute_249.json" # Path where inference results will be saved
rouge = Rouge(metrics=['rouge-1'])
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


def parse_model_output(text):
    if text is None:
        text = ""
    role_match = re.search(r"Role:\s*\[?([^\]\n,]+)\]?", text, re.IGNORECASE)
    name_match = re.search(r"Name:\s*\[?([^\]\n,]+)\]?", text, re.IGNORECASE)
    
    role = role_match.group(1).strip() if role_match else None
    name = name_match.group(1).strip() if name_match else None
    
    return role, name

# ===== Asynchronously Process Single Sample =====
async def process_item(index, item, sem, claude_client_instance, stats):
    async with sem:
        image_path = item["images"][0]
        gt_response = item["messages"][-1]["content"]

        # 加载并编码图像
        async with aiofiles.open(image_path, "rb") as f:
            content = await f.read()
        encoded_image = base64.b64encode(content).decode("utf-8")
        image_data_uri = f"data:image/png;base64,{encoded_image}"

        # 构造消息
        messages =[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_data_uri}},
                            {
                                "type": "text",
                                "text": (
                                        '''
                                        You are viewing a screenshot of a webpage where a specific element is marked with a red box. \nYour task is to predict its ARIA role and accessible name based on the context of the webpage and the visual appearance of the element. Please make your prediction using the following list of roles with their semantic descriptions, combined with visual clues from the screenshot (such as text, position, and style).\n** Possible Roles and Their Semantics\n1.link: A hyperlink used to navigate to other pages or resources.\n2.button: A button used to trigger actions (e.g., submit, confirm).\n3.textbox: A single-line text input field for entering free text.\n4.searchbox: A search input field for entering search queries.\n5.checkbox: A checkbox for multiple-choice options.\n6.radio: A radio button for single-choice options.\n7.slider: A slider for adjusting a range of values.\n8.spinbutton: A numeric adjuster for incrementing or decrementing values.\n9.combobox: A dropdown selection box allowing choices from options.\n10.option: A single option, typically within a dropdown or list.\n11.listbox: A list selection box displaying multiple selectable items.\n12.img: An image used to display visual content.\n13.form: A form containing a collection of user input controls.\n14.navigation: A navigation area providing links for the page or site.\n16.banner: A header, typically containing the site title or banner.\n17.contentinfo: A footer, usually containing copyright or contact information.\n18.article: An article, an independent content block (e.g., news, post).\n19.search: A search area, typically containing search functionality.\n20.heading: A heading used for content hierarchy (level may need to be inferred, e.g., heading level 1).\n21.list: A list containing multiple items.\n22.listitem: A list item, a single entry within a list.\n23.table: A table for displaying data in rows and columns.\n24.row: A table row containing cells.\n25.columnheader: A column header in a table.\n26.rowheader: A row header in a table.\n27.cell: A cell, a data item in a table.\n28.dialog: A dialog box, such as a popup window or modal.\n29.progressbar: A progress bar showing task progress.\n30.status: A status update providing dynamic information.\n31.paragraph: A paragraph, a block of text content.\n\n** Prediction Guidance\n1. Role:\nSelect the most matching role based on the element\u2019s visual characteristics (e.g., button shape, input field border) and context (e.g., located in a navigation bar or form).\nRefer to the role semantics to ensure the prediction aligns with its definition.\nIf uncertain, prioritize values from the above role list and avoid arbitrary guesses.\n2. Name:\nExtract the name from visible text on the element (e.g., \u201cSubmit\u201d on a button, a label next to an input field).\nIf no visible text is present, infer a reasonable name (e.g., \u201cunlabeled button\u201d).\n** Output Format\nPlease provide your prediction in the following format:\nRole: [role], Name: [name]\nRemeber the [] where  [role] and [name] MUST be enclosed in square brackets []
                                        '''
                                ),
                            },
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
        await asyncio.sleep(15) # 暂停0.5秒。根据你的RPS限制调整这个值。
                                 # 例如，如果RPS是2，你可能需要等待0.5秒。
        # 解析 Role 和 Name
        pred_role, pred_name = parse_model_output(pred_text)
        gt_role, gt_name = parse_model_output(gt_response)
        
        # Role 匹配评估
        match_role = pred_role == gt_role

        # Name 匹配评估 (使用 ROUGE-1 F1 分数)
        # 按照示例，处理空字符串以避免 ROUGE 库出错
        pred_for_rouge = pred_name if pred_name else " "
        gt_for_rouge = gt_name if gt_name else " "
        
        try:
            # 将 gt 包装成列表，以匹配 ROUGE 库的输入格式
            scores = rouge.get_scores([pred_for_rouge], [gt_for_rouge], avg=True)
            name_f1 = scores['rouge-1']['f']
            name_precision = scores['rouge-1']['p']
            name_recall = scores['rouge-1']['r']
        except Exception:
            # 如果 ROUGE 计算出现任何异常，则该样本分数为0
            name_f1, name_precision, name_recall = 0.0, 0.0, 0.0

        match_name = name_f1 == 1.0

        match_all = match_role and match_name

        # 统计信息更新
        stats["total"] += 1
        stats["role_correct"] += int(match_role)
        stats["all_correct"] += int(match_all)
        stats["name_f1_total"] += name_f1
        stats["name_precision_total"] += name_precision
        stats["name_recall_total"] += name_recall


        # 每 N 步打印一次准确率
        if stats["total"] % ACCURACY_PRINT_INTERVAL == 0:
            role_acc = stats["role_correct"] / stats["total"] * 100
            avg_name_f1 = stats["name_f1_total"] / stats["total"] * 100
            full_acc = stats["all_correct"] / stats["total"] * 100
            print(f"\n📊 Step {stats['total']}: Role Acc={role_acc:.2f}%, Avg Name ROUGE-1 F1={avg_name_f1:.2f}%, Full Score={(role_acc + avg_name_f1) / 2:.2f}%\n")

        return {
            "image": os.path.basename(image_path),
            "ground_truth": {"role": gt_role, "name": gt_name},
            "prediction": {"role": pred_role, "name": pred_name},
            "metrics_per_sample": {
                "name_rouge1_f1": name_f1,
                "name_rouge1_precision": name_precision,
                "name_rouge1_recall": name_recall
            },
            "match_role": match_role,
            "match_name": match_name,
            "match_all": match_all,
            "raw":pred_text
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
    stats = {
        "total": 0, 
        "role_correct": 0, 
        "all_correct": 0,
        "name_f1_total": 0.0,
        "name_precision_total": 0.0,
        "name_recall_total": 0.0
    }

    # Create tasks for each item
    tasks = [process_item(i, item, sem, claude_client, stats) for i, item in enumerate(test_data)]

    print(f"\n🚀 Starting evaluation of {len(tasks)} samples using Claude on AWS Bedrock...\n")
    results = await tqdm_asyncio.gather(*tasks)

    total_samples = stats["total"] if stats["total"] > 0 else 1
    
    # 计算最终指标
    role_acc = (stats["role_correct"] / total_samples * 100)
    full_acc = (stats["all_correct"] / total_samples * 100)
    
    # 计算 Name 的平均 Precision, Recall, ROUGE-1 F1-Score
    avg_name_precision = (stats["name_precision_total"] / total_samples * 100)
    avg_name_recall = (stats["name_recall_total"] / total_samples * 100)
    avg_name_f1 = (stats["name_f1_total"] / total_samples * 100)

    # 错误样例收集
    errors = [r for r in results if not r["match_all"]]

    # 写入 JSON 文件
    output = {
        "metrics": {
            "total_samples": stats["total"],
            "role_accuracy": role_acc,
            "name_metrics": {
                "average_rouge1_precision": avg_name_precision,
                "average_rouge1_recall": avg_name_recall,
                "average_rouge1_f1_score": avg_name_f1,
            },
            "full_match_accuracy": full_acc,
            "full_score":(role_acc + avg_name_f1) / 2
        },
        "errors": errors
    }

    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # 控制台输出准确率
    print(f"\n✅ Evaluation Complete")
    print(f"🎯 Role Accuracy             : {role_acc:.2f}%")
    print(f"🎯 Avg Name ROUGE-1 Precision: {avg_name_precision:.2f}%")
    print(f"🎯 Avg Name ROUGE-1 Recall   : {avg_name_recall:.2f}%")
    print(f"🎯 Avg Name ROUGE-1 F1-Score : {avg_name_f1:.2f}%")
    print(f"🎯 Full Match Accuracy       : {full_acc:.2f}%")
    print(f"🎯 Full Score                : {(role_acc + avg_name_f1) / 2:.2f}%")
    print(f"📁 Results saved to: {OUTPUT_JSON_PATH}")

    # 错误示例展示
    print("\n❌ Sample Errors (up to 5):")
    for r in errors[:5]:
        print(f"- Image        : {r['image']}")
        print(f"  Ground Truth : {r['ground_truth']}")
        print(f"  Prediction   : {r['prediction']}")
        print(f"  Name ROUGE-1 F1: {r['metrics_per_sample']['name_rouge1_f1']:.2f}\n")
# ===== 执行主函数 =====
if __name__ == "__main__":
    asyncio.run(main())