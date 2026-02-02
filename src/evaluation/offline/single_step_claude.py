import os
import sys
import json
import base64
import re
import asyncio
import aiofiles
from tqdm.asyncio import tqdm_asyncio
import boto3

# ===================================================================
# ===== 1. Configuration =====
# ===================================================================

# --- Task Configuration ---
TEST_JSON_PATH = "path/to/your/test.json"  # TODO: Update this path
MAX_SAMPLE = 70
MAX_CONCURRENT_REQUESTS = 1
ACCURACY_PRINT_INTERVAL = 10

# --- Model and Output Configuration ---
Model_name = "Claude"
BEDROCK_MODEL_ID = "us.anthropic.claude-sonnet-4-20250514-v1:0"
OUTPUT_JSON_PATH = f"./results/Test-{Model_name}-Single_Step.json"

# --- AWS Bedrock Credentials (from environment variables) ---
# Set these environment variables before running:
# export AWS_ACCESS_KEY_ID="your_access_key"
# export AWS_SECRET_ACCESS_KEY="your_secret_key"
# export AWS_REGION_NAME="us-east-1"
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_REGION_NAME = os.getenv("AWS_REGION_NAME", "us-east-1")

# ===================================================================
# ===== 2. AWS Bedrock Claude Client =====
# ===================================================================

class BedrockClaudeClient:
    """
    Client for interacting with Claude models on AWS Bedrock.
    """
    def __init__(self, access_key, secret_key, region_name, model_id):
        self.model_id = model_id
        try:
            self.bedrock_client = boto3.client(
                service_name='bedrock-runtime',
                region_name=region_name,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key
            )
            print(f"Boto3 client created successfully, region: '{region_name}', model: '{self.model_id}'!")
        except Exception as e:
            raise ConnectionError(f"Failed to create Bedrock client: {e}. Please check your AWS credentials and region name.")

    def _parse_data_url(self, data_url):
        if not data_url.startswith("data:"): return None, None
        parts = data_url.split(',', 1)
        if len(parts) < 2: return None, None
        metadata = parts[0][len("data:"):].split(';')
        media_type = metadata[0]
        base64_data = parts[1]
        if "base64" not in metadata: return None, None
        return base64_data, media_type

    def chat(self, messages, max_tokens=2048, temperature=0.1):
        """
        Send messages to the Claude model and get a response.
        This function is compatible with OpenAI format message lists.
        """
        if not hasattr(self, 'bedrock_client'):
            raise RuntimeError("Bedrock client not successfully initialized.")

        claude_system_message = None
        claude_messages_payload = []

        for openai_msg in messages:
            role = openai_msg.get("role")
            content = openai_msg.get("content")

            if role == "system":
                if isinstance(content, str):
                    claude_system_message = content
                else:
                     print(f"Warning: Unsupported system content type: {type(content)}")
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
            raise ValueError("No valid 'user' or 'assistant' messages to send to Claude after conversion.")

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
            raise RuntimeError(f"Error calling Claude model: {error_message}")


# ===================================================================
# ===== 3. Core Functions =====
# ===================================================================

def extract_action(text: str):
    """Extract standardized action string from model output."""
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
    """Parse Ground Truth string that may contain multiple answers."""
    action_parts = gt_content.split(';')
    parsed_actions = [extract_action(part.strip()) for part in action_parts]
    return [action for action in parsed_actions if action is not None]

def compare_actions(prediction: str, ground_truth_list: list) -> bool:
    """Smart comparison function, specially handling TYPE actions."""
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
# ===== 4. Async Processing Functions =====
# ===================================================================

async def process_item(index, item, sem, claude_client, stats):
    """Process a single item using Claude client."""
    async with sem:
        image_paths = item["images"]
        prompt = item["messages"][0]["content"]
        gt_json_str = item["messages"][-1]["content"]
        gt_answers_list = parse_ground_truth(gt_json_str)

        if not gt_answers_list:
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
                return {"match": False, "raw_model_output": f"Image not found: {path}"}

        messages = [
            {
                "role": "user",
                "content": image_contents + [
                    {
                        "type": "text",
                        "text": "Based on the provided image, task description, and AxTree, please output the element IDs required to complete the task. The expected answer format is: `Action: click [AxTree ID]`. " + prompt.strip(),
                    }
                ],
            },
        ]

        pred_text = ""
        try:
            response_data = await asyncio.to_thread(
                claude_client.chat,
                messages=messages,
                temperature=0.1,
                max_tokens=2048
            )
            pred_text = response_data['response_text'].strip()
        except Exception as e:
            pred_text = f"[ERROR] {str(e)}"

        pred_answer = extract_action(pred_text)
        match = compare_actions(pred_answer, gt_answers_list)

        stats["total"] += 1
        stats["correct"] += int(match)
        if stats["total"] > 0 and stats["total"] % ACCURACY_PRINT_INTERVAL == 0:
            acc = stats["correct"] / stats["total"] * 100
            print(f"\nStep {stats['total']}: Accuracy = {acc:.2f}%\n")

        return {
            "images": image_paths,
            "prompt": prompt,
            "ground_truth": ";".join(gt_answers_list),
            "prediction": pred_answer,
            "match": match,
            "raw_model_output": pred_text
        }

# ===================================================================
# ===== 5. Main Function =====
# ===================================================================

async def main():
    # Check for required environment variables
    if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
        print("Error: Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables.")
        sys.exit(1)

    try:
        claude_client = BedrockClaudeClient(
            access_key=AWS_ACCESS_KEY_ID,
            secret_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION_NAME,
            model_id=BEDROCK_MODEL_ID
        )
    except Exception as e:
        print(f"Failed to initialize Bedrock client: {e}")
        sys.exit(1)

    with open(TEST_JSON_PATH, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    if MAX_SAMPLE is not None and MAX_SAMPLE > 0:
        test_data = test_data[:MAX_SAMPLE]

    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    stats = {"total": 0, "correct": 0}

    tasks = [process_item(i, item, sem, claude_client, stats) for i, item in enumerate(test_data)]

    print(f"\nStarting evaluation of {len(tasks)} samples using model '{BEDROCK_MODEL_ID}'...\n")
    results = await tqdm_asyncio.gather(*tasks)

    valid_results = [r for r in results if r is not None]
    if not valid_results:
        print("\nNo valid samples processed. Evaluation cannot be completed.")
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

    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nEvaluation complete")
    print(f"Accuracy: {accuracy:.2f}% ({final_correct}/{final_total})")
    print(f"Results saved to: {OUTPUT_JSON_PATH}")

    if errors:
        print("\nError samples (up to 5):")
        for r in errors[:5]:
            print(f"- Images       : {', '.join(r['images'])}")
            print(f"  Ground Truth : {r['ground_truth']}")
            print(f"  Prediction   : {r['prediction']}")
            raw_output_snippet = r['raw_model_output'].replace('\n', ' ')
            if len(raw_output_snippet) > 200:
                raw_output_snippet = "..." + raw_output_snippet[-200:]
            print(f"  Raw Output   : {raw_output_snippet}\n")

# ===================================================================
# ===== 6. Entry Point =====
# ===================================================================

if __name__ == "__main__":
    asyncio.run(main())
