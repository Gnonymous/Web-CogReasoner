import os
import sys
import json
import base64
import re
import asyncio
import aiofiles
from tqdm.asyncio import tqdm_asyncio # Used for progress bar in async tasks
import boto3 # Import boto3 for AWS Bedrock interaction

Test_Model = "Claude" # Define the model name for testing

# ===== Configuration Items =====
TEST_JSON_PATH = "/code/CogReasoner/Test/VisualWebBench_Action_Ground_103.json" # Path to the test set JSON file
MODEL_NAME = "us.anthropic.claude-sonnet-4-20250514-v1:0" # Specify the Claude model name for inference
MAX_SAMPLE = 103 # Maximum number of samples to test
MAX_CONCURRENT_REQUESTS = 1 # Maximum concurrent requests (set to 10 for async processing)
ACCURACY_PRINT_INTERVAL = 10 # Print current accuracy after processing this many samples
OUTPUT_JSON_PATH = f"/code/CogReasoner/Code/Evalaute/Result/Test-{Test_Model}-VisualWebBench_Action_Ground_103.json" # Path where inference results will be saved

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


# ===== Extract Answer Letter from Model Output =====
def extract_answer_letter(text):
    """
    Extracts the answer option (A, B, C, etc.) from the model's text output.
    Attempts multiple pattern matches for robustness.
    """
    # Prioritize matching structures with '### Final Choice', e.g., "### Final Choice: Option A"
    match = re.search(r"###\s*Final Choice:\s*Option[:\s]*([A-H])\b", text, re.IGNORECASE)
    if match:
        return match.group(1).upper() # Extract and convert to uppercase

    # Next, match 'The answer is: X' or 'The answer is X' structures
    match = re.search(r"The answer is[:\s]*([A-H])\b", text, re.IGNORECASE)
    if match:
        return match.group(1).upper() # Extract and convert to uppercase

    # Fallback match: find isolated uppercase letters (A to H) in the text
    # findall returns all matches; we take the last one as the model usually gives the answer at the end
    fallback = re.findall(r"\b([A-H])\b", text.upper())
    if fallback:
        return fallback[-1] # Return the last matched letter

    return None # Return None if no match is found

# ===== Asynchronously Process Single Sample =====
async def process_item(index, item, sem, claude_client_instance, stats):
    async with sem: # Control concurrency with semaphore
        image_path = item["images"][0] # Get image path
        gt_answer = item["messages"][-1]["content"].strip().upper() # Extract ground truth answer, convert to uppercase
        prompt = item["messages"][0]["content"] # Extract user prompt

        # Encode image to Base64 string
        async with aiofiles.open(image_path, "rb") as f:
            content = await f.read() # Asynchronously read image content
        encoded_image = base64.b64encode(content).decode("utf-8") # Base64 encode
        image_data_uri = f"data:image/png;base64,{encoded_image}" # Construct Data URI format

        pred_text = "" # Initialize model prediction text
        try:
            # Send inference request to the model
            # The ClaudeClient.chat method is synchronous, so we run it in a thread pool
            response_data = await asyncio.to_thread(
                claude_client_instance.chat,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."}, # System message
                    {
                        "role": "user",
                        "content": [
                            # BedrockClaudeClient expects image_url format for Data URI
                            {"type": "image_url", "image_url": {"url": image_data_uri}}, 
                            {
                                "type": "text",
                                "text": prompt + "You should directly tell me your choice in a single uppercase letter, and do not output any explanation or any other contents.", # Text prompt
                            },
                        ],
                    },
                ],
                max_tokens=2048, # Limit max length of generated text
                temperature=0.1 # Control randomness of generated text
            )
            pred_text = response_data['response_text'].strip() # Extract model generated text content
        except Exception as e:
            pred_text = f"[ERROR] {str(e)}" # Capture exception and log error message
        await asyncio.sleep(10) # 暂停0.5秒。根据你的RPS限制调整这个值。
        pred_answer = extract_answer_letter(pred_text) # Extract predicted answer from model output
        match = pred_answer == gt_answer # Check if predicted answer matches ground truth

        stats["total"] += 1 # Increment total processed samples
        stats["correct"] += int(match) # Increment correct count if match

        # Print current accuracy periodically
        if stats["total"] % ACCURACY_PRINT_INTERVAL == 0:
            acc = stats["correct"] / stats["total"] * 100
            print(f"\n📊 Step {stats['total']}: Accuracy = {acc:.2f}%\n")

        return {
            "image": image_path, # Original image path
            "ground_truth": gt_answer, # Ground truth answer
            "prediction": pred_answer, # Model predicted answer
            "match": match, # Whether it matched
            "raw_model_output": pred_text # Raw model output text
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

    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS) # Create semaphore for concurrency control
    stats = {"total": 0, "correct": 0} # Initialize statistics

    # Create tasks for each item
    tasks = [process_item(i, item, sem, claude_client, stats) for i, item in enumerate(test_data)]

    print(f"\n🚀 Starting evaluation of {len(tasks)} samples using Claude on AWS Bedrock...\n")
    results = await tqdm_asyncio.gather(*tasks) # Execute all tasks concurrently with a progress bar

    accuracy = stats["correct"] / stats["total"] * 100 # Calculate final accuracy
    errors = [r for r in results if not r["match"]] # Filter out all non-matching (incorrect) samples

    # Prepare JSON structure for output
    output = {
        "metrics": {
            "total": stats["total"], # Total samples
            "correct": stats["correct"], # Correct samples
            "accuracy": accuracy # Final accuracy
        },
        "errors": errors # Detailed info of incorrect samples
    }

    # Write results to the specified JSON file
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False) # Save JSON in a readable format, ensure Chinese support

    # Print evaluation summary to console
    print(f"\n✅ Evaluation Complete")
    print(f"🎯 Accuracy: {accuracy:.2f}%")
    print(f"📁 Results saved to: {OUTPUT_JSON_PATH}")

    print("\n❌ Sample Errors (up to 5):")
    # Print detailed information for the first 5 error samples
    for r in errors[:5]:
        print(f"- Image          : {r['image']}")
        print(f"  Ground Truth   : {r['ground_truth']}")
        print(f"  Prediction     : {r['prediction']}")
        print(f"  Raw Output     : {r['raw_model_output']}\n")

# ===== Entry Point =====
if __name__ == "__main__":
    asyncio.run(main()) # Run the main asynchronous function
    sys.exit(0) # Force exit to prevent the async event loop from hanging