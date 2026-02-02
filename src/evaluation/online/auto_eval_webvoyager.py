import argparse
import os
import json
import time
import re
import base64
import google.generativeai as genai
from PIL import Image
from openai import OpenAI


SYSTEM_PROMPT = """As an evaluator, you will be presented with three primary components to assist you in your role:

1. Web Task Instruction: This is a clear and specific directive provided in natural language, detailing the online activity to be carried out.
2. Result Screenshots: This is a visual representation of the screen showing the result or intermediate state of performing a web task.
3. Result Response: This is a textual response obtained after the execution of the web task.

-- You DO NOT NEED to interact with web pages.
-- You SHOULD NOT make assumptions based on information not present in the screenshot when comparing it to the instructions.

-- Your primary responsibility is to assess whether the **main goal of the task was successfully accomplished**.
-- **Minor inaccuracies, formatting differences, or reasonable extrapolations** in the final response should be tolerated as long as the core information requested in the instruction has been correctly identified and presented. For example, if the task is to find synonyms and the agent correctly lists the main ones but also includes a grammatical variation seen in an example sentence, this should be considered a success.

-- If the instruction involves multiple distinct sub-tasks (e.g., "find the price AND check the delivery time"), the agent must attempt to complete all sub-tasks. If the primary objective of each sub-task is met, the overall task is successful.

-- Regarding discrepancies between the screenshot and the response:
-- 1) If the Result Response **fundamentally contradicts** a key piece of information in the screenshot (e.g., price is $50 vs $100), the screenshot prevails and the task is a failure.
-- 2) If the Result Response **augments or slightly misinterprets** content from the screenshot (like the synonym example above), this is acceptable and does not constitute a failure.
-- 3) If the Result Response contains information **not visible** in the final screenshot, you should generally trust it, as the agent may have gathered it from previous steps.

You should elaborate on how you arrived at your final evaluation and then provide a definitive verdict on whether the task has been successfully accomplished, either as 'SUCCESS' or 'NOT SUCCESS'.
"""

USER_PROMPT = """TASK: <task>
Result Response: <answer>
<num> screenshots at the end: """

def encode_image(image_path):
    """Encodes an image to a base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def auto_eval(process_dir, client, api_model, img_num, provider):
    """Evaluates a single task directory using the specified LLM."""
    print(f'--------------------- {process_dir} (Provider: {provider.upper()}) ---------------------')
    res_files = sorted(os.listdir(process_dir))
    
    try:
        with open(os.path.join(process_dir, 'interact_messages.json')) as fr:
            it_messages = json.load(fr)
    except FileNotFoundError:
        print(f" interact_messages.json not found in {process_dir}. Skipping.")
        return {"success": 0, "reasoning": "interact_messages.json not found."}

    if len(it_messages) < 2:
        print('Not find answer for ' + process_dir + ', not enough messages.')
        print()
        return {"success": 0, "reasoning": "Not enough messages to find an answer."}

    # Extract Task Instruction
    first_user_message = None
    for msg in it_messages:
        if msg.get("role") == "user":
            first_user_message = msg
            break

    if not first_user_message:
        print(f"Could not find a 'user' role message in {process_dir}. Skipping.")
        return {"success": 0, "reasoning": "Could not find a 'user' role message."}

    content_list = first_user_message.get("content")
    if not isinstance(content_list, list) or not content_list:
        print(f"User message content is not a valid list in {process_dir}. Skipping.")
        return {"success": 0, "reasoning": "User message content is not a valid list."}

    task_text_block = ""
    for item in content_list:
        if item.get("type") == "text":
            task_text_block = item.get("text", "")
            break

    if not task_text_block.strip().startswith("Task:"):
        print(f"Could not find 'Task:' prefix in the first text block of the user message in {process_dir}. Skipping.")
        return {"success": 0, "reasoning": "Could not find 'Task:' prefix."}
    
    match = re.search(r"Task:(.*?)(?:\nOBSERVATION:|\Z)", task_text_block, re.DOTALL)
    if not match:
        print(f"Regex failed to extract task content from text block in {process_dir}.")
        return {"success": 0, "reasoning": "Regex failed to extract task content."}
    
    task_content = match.group(1).strip()
    print(f"Extracted Task: {task_content}")

    # 2. Extract Agent's Final Answer
    ans_info = it_messages[-1]["content"]
    
    # Check for valid termination actions
        print('Not find answer for ' + process_dir)
        print()
        return {"success": 0, "reasoning": "Final message does not contain 'Action: stop' or 'Action: ANSWER'."}
    
    pattern_ans = r"Action: (?:stop|ANSWER)\s*\[(.*?)\]"
    matches_ans = re.search(pattern_ans, ans_info, re.DOTALL)
    
    if not matches_ans:
        print(f"Found 'Action: stop/ANSWER' but could not extract content from brackets for {process_dir}.")
        return {"success": 0, "reasoning": "Could not extract content from stop/ANSWER action."}
        
    answer_content = matches_ans.group(1).strip()
    print(f"Extracted Answer: {answer_content}")

    # Retrieve and Filter Screenshots
    pattern_png = r'screenshot(\d+)\.png'
    png_files = [(filename, int(re.search(pattern_png, filename).group(1))) for filename in res_files if re.search(pattern_png, filename)]
    if not png_files:
        print(f"No screenshots found in {process_dir}. Skipping.")
        return {"success": 0, "reasoning": "No screenshots found."}
    png_files.sort(key=lambda x: x[1])
    end_files = png_files[-img_num:]
    
    # 4. Construct User Prompt
    user_prompt_tmp = USER_PROMPT.replace('<task>', task_content)
    user_prompt_tmp = user_prompt_tmp.replace('<answer>', answer_content)
    user_prompt_tmp = user_prompt_tmp.replace('<num>', str(len(end_files)))

    eval_response_text = ""
    
    # Call LLM API
    if provider == 'openai':
        whole_content_img = [{'type': 'image_url', 'image_url': {"url": f"data:image/png;base64,{encode_image(os.path.join(process_dir, png_file))}"}} for png_file, _ in end_files]
        messages = [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': [{'type': 'text', 'text': user_prompt_tmp}] + whole_content_img + [{'type': 'text', 'text': "Your verdict:\n"}]}]
        
        while True:
            try:
                print('Calling OpenAI API to get the auto evaluation......')
                openai_response = client.chat.completions.create(model=api_model, messages=messages, max_tokens=1000, seed=42, temperature=0)
                eval_response_text = openai_response.choices[0].message.content
                break
            except Exception as e:
                print(e)
                if 'RateLimitError' in type(e).__name__: time.sleep(10)
                elif 'APIError' in type(e).__name__: time.sleep(15)
                elif 'InvalidRequestError' in type(e).__name__ or (hasattr(e, 'code') and e.code == 'content_filter'):
                    return {"success": 0, "reasoning": f'API Error: {e}'}
                else: time.sleep(10)

    elif provider == 'gemini':
        full_prompt_text = SYSTEM_PROMPT + "\n\n---\n\n" + user_prompt_tmp
        prompt_parts = [full_prompt_text]
        for png_file, _ in end_files:
            prompt_parts.append(Image.open(os.path.join(process_dir, png_file)))
        prompt_parts.append("\nYour verdict:\n")
        
        model = client
        while True:
            try:
                print('Calling Gemini API to get the auto evaluation......')
                gemini_response = model.generate_content(prompt_parts, generation_config=genai.types.GenerationConfig(temperature=0.0))
                if gemini_response.prompt_feedback.block_reason.name != "BLOCK_REASON_UNSPECIFIED":
                    return {"success": 0, "reasoning": f"Content filter for Gemini! Reason: {gemini_response.prompt_feedback.block_reason.name}"}
                eval_response_text = gemini_response.text
                break
            except Exception as e:
                print(e)
                time.sleep(10)

    # Process API Response
    print('API call complete...')
    print(eval_response_text)

    evaluation_result = {
        "reasoning": eval_response_text,
        "success": 1 if 'SUCCESS' in eval_response_text and 'NOT SUCCESS' not in eval_response_text else 0
    }
    
    print('Auto_eval_res:', evaluation_result["success"])
    print()
        
    return evaluation_result

def main():
    print("Starting evaluation with the following parameters:")

    parser = argparse.ArgumentParser()
    parser.add_argument('--process_dir', type=str, required=True, help='Path to the process directory containing task results.')
    parser.add_argument('--provider', type=str, default='gemini', choices=['openai', 'gemini'], help="The provider for the evaluation model.")
    parser.add_argument("--openai_api_key", default=os.getenv("OPENAI_API_KEY"), type=str, help="Your OpenAI API key.")
    parser.add_argument("--gemini_api_key", default=os.getenv("GEMINI_API_KEY"), type=str, help="Your Google Gemini API key.")
    parser.add_argument("--api_model", default="gemini-2.5-pro-preview-06-05", type=str, help="API model name. E.g., 'gpt-4o' for OpenAI, 'gemini-2.5-pro-latest' for Gemini.")
    parser.add_argument("--max_attached_imgs", type=int, default=15)
    parser.add_argument('--output_report_path', type=str,required=True, help="Path to save the final evaluation report.")
    parser.add_argument('--force-reeval',action='store_true',help="Force re-evaluation of all tasks. Clears processed list and re-runs LLM evaluation.")   

    args = parser.parse_args()
    print("Starting evaluation with the following parameters:")
    print(f"  - Process Directory: {args.process_dir}")
    print(f"  - Provider: {args.provider}")
    print(f"  - Model: {args.api_model}")
    print(f"  - Force Re-evaluation: {args.force_reeval}")
    print(f"  - Output Report: {args.output_report_path}")

    # Initialize API Client
    client = None
    if args.provider == 'openai':
        if args.openai_api_key == 'key': raise ValueError("Please provide an OpenAI API key.")
        client = OpenAI(api_key=args.openai_api_key)
        print("Using OpenAI for evaluation.")
    elif args.provider == 'gemini':
        if args.gemini_api_key == 'key': raise ValueError("Please provide a Gemini API key.")
        genai.configure(api_key=args.gemini_api_key)
        client = genai.GenerativeModel(args.api_model)
        print("Using Gemini for evaluation.")

    webs = ['Allrecipes', 'Amazon', 'Apple', 'ArXiv', 'BBC News', 'Booking', 'Cambridge Dictionary',
            'Coursera', 'ESPN', 'GitHub', 'Google Flights', 'Google Map', 'Google Search', 'Huggingface', 'Wolfram Alpha']

    # Main Evaluation Loop

    overall_results = {}
    # Load existing report
    if os.path.exists(args.output_report_path) and not args.force_reeval:
        print(f"Found existing summary report at {args.output_report_path}. Loading it.")
        try:
            with open(args.output_report_path, 'r') as f:
                overall_results = json.load(f)
            # Validate format
            if "processed_tasks" not in overall_results or not isinstance(overall_results["processed_tasks"], list):
                print("   - WARNING: Existing report is in an old or corrupted format. Starting fresh to ensure consistency.")
                overall_results = {}
        except (json.JSONDecodeError, IOError) as e:
            print(f"   - WARNING: Could not read or parse existing report. Starting fresh. Error: {e}")
            overall_results = {}
    
    # Initialize report structure
    if not overall_results:
        print("Initializing a new summary report.")
        overall_results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": args.api_model,
            "provider": args.provider,
            "processed_tasks": [], # Tracks processed tasks to avoid duplicates
            "total_tasks": 0,
            "total_success": 0,
            "total_failure": 0,
            "success_rate": 0.0,
            "by_website": {},
            "failed_tasks_reasoning": []
        }
    else:
        # Reset stats if force re-evaluation
        if args.force_reeval:
            print("Force re-evaluation activated. Clearing aggregate statistics (total_tasks, success_rate, etc.). Individual task eval files will be re-generated if they don't exist or if --force-reeval was true during their generation.")
            overall_results["total_tasks"] = 0
            overall_results["total_success"] = 0
            overall_results["total_failure"] = 0
            overall_results["success_rate"] = 0.0
            overall_results["by_website"] = {}
            overall_results["failed_tasks_reasoning"] = []

            overall_results = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model": args.api_model,
                "provider": args.provider,
                "processed_tasks": [],
                "total_tasks": 0,
                "total_success": 0,
                "total_failure": 0,
                "success_rate": 0.0,
                "by_website": {},
                "failed_tasks_reasoning": []
            }


    # Track processed tasks
    current_run_processed_tasks = set(overall_results.get('processed_tasks', []))
    
    for web in webs:
        # Ensure website key exists
        if web not in overall_results["by_website"]:
            overall_results["by_website"][web] = {"count": 0, "success_count": 0, "success_rate": 0.0}

        for idx in range(0, 46): # Assuming 0-45 tasks per website
            task_id = f'{web}--{idx}'
            file_dir = os.path.join(args.process_dir, f'task{web}--{idx}')

            if not os.path.exists(file_dir):
                continue
            
            if task_id in current_run_processed_tasks:
                print(f"--> Task {task_id} has already been aggregated in this run. Skipping re-aggregation.")
                continue

            print(f"\n>>> Processing task {task_id} in directory {file_dir}")

            eval_file_path = os.path.join(file_dir, f'eval_res_structured_{args.provider}.json')
            evaluation_data = None
            
            if os.path.exists(eval_file_path) and not args.force_reeval:
                print(f"    - Existing individual report found. Loading result from '{eval_file_path}'...")
                try:
                    with open(eval_file_path, 'r') as f:
                        evaluation_data = json.load(f)
                    if "success" in evaluation_data and "reasoning" in evaluation_data:
                        print(f"    - Loaded successfully. Success: {evaluation_data.get('success', 'N/A')}. Skipping API call.")
                    else:
                        print(f"    - WARNING: Individual report '{eval_file_path}' is incomplete or malformed. Will re-evaluate.")
                        evaluation_data = None 
                except (json.JSONDecodeError, IOError) as e:
                    print(f"    - WARNING: Could not read or parse individual report '{eval_file_path}'. Will re-evaluate. Error: {e}")
                    evaluation_data = None 
            
            # Run LLM Evaluation if needed
            if evaluation_data is None:
                print(f"    - Running LLM evaluation for {task_id}...")
                evaluation_data = auto_eval(file_dir, client, args.api_model, args.max_attached_imgs, args.provider)
                
                # Save individual evaluation report
                if evaluation_data and "success" in evaluation_data and "reasoning" in evaluation_data:
                    try:
                        with open(eval_file_path, 'w') as fw:
                            json.dump(evaluation_data, fw, indent=2)
                        print(f"    - New individual report saved to '{eval_file_path}'")
                    except IOError as e:
                        print(f"    - CRITICAL: Failed to write individual report for {task_id}! Error: {e}")
                else:
                    print(f"    - LLM evaluation for {task_id} returned invalid data. Skipping aggregation.")
                    continue # Skip invalid result

            # Update report
            overall_results["total_tasks"] += 1
            overall_results["by_website"][web]["count"] += 1

            if evaluation_data['success'] == 1:
                overall_results["total_success"] += 1
                overall_results["by_website"][web]["success_count"] += 1
            else:
                overall_results["total_failure"] += 1
                overall_results["failed_tasks_reasoning"].append({
                    "task_id": task_id,
                    "reasoning": evaluation_data.get("reasoning", "No reasoning provided.")
                })

            # Mark as processed
            current_run_processed_tasks.add(task_id)
            overall_results['processed_tasks'] = list(current_run_processed_tasks)

            # Recalculate metrics
            if overall_results["total_tasks"] > 0:
                overall_results["success_rate"] = overall_results["total_success"] / overall_results["total_tasks"]
            if overall_results["by_website"][web]["count"] > 0:
                overall_results["by_website"][web]["success_rate"] = overall_results["by_website"][web]["success_count"] / overall_results["by_website"][web]["count"]
            
            # Save summary report
            try:
                overall_results["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
                with open(args.output_report_path, 'w') as f:
                    json.dump(overall_results, f, indent=4)
                print(f"    - Real-time summary report '{args.output_report_path}' has been updated.")
            except IOError as e:
                print(f"    - WARNING: Failed to write real-time summary! Error: {e}")
    


    # Final Summary
    print("\n\n=====================================================")
    print(f"      Final Evaluation Summary Report            ")
    print("=====================================================")
    print(f"Total tasks processed: {overall_results['total_tasks']}")
    print(f"Overall Success Rate: {overall_results['success_rate']:.2%}")
    print(f"\nTotal failures: {overall_results['total_failure']}")
    if overall_results['total_failure'] > 0:
        print("\n--- Details of Failed Tasks (first 5) ---")
        for i, failure in enumerate(overall_results.get("failed_tasks_reasoning", [])[:5]):
            print(f"{i+1}. Task ID: {failure['task_id']}")
            
    print(f"\nFull report saved to: {args.output_report_path}")

if __name__ == "__main__":
    main()