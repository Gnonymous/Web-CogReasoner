"""Auto-eval script for Mind2Web Cross-Web using OpenAI/Gemini."""
import argparse
import os
import json
import time
import re
import base64
import google.generativeai as genai
from PIL import Image
from openai import OpenAI


# --- Constant Prompts ---
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

# User Prompt Templates
USER_PROMPT_WITH_ANSWER = """TASK: <task>
Result Response: <answer>
<num> screenshots at the end: """

USER_PROMPT_NO_ANSWER = """TASK: <task>
The agent did not provide a definitive textual "Result Response". Please evaluate the task based solely on the provided "Web Task Instruction" and the following <num> screenshots, which represent the final state of the web browsing.
Your primary goal is to determine if the main objective of the task, as described in the "Web Task Instruction", appears to have been completed in the screenshots.
<num> screenshots at the end: """


def encode_image_base64(image_path):
    """Encodes an image to a base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def auto_eval(process_dir, client, api_model, img_num, provider):
    """Evaluates a single task directory using the specified LLM."""
    print(f'--------------------- {process_dir} (Provider: {provider.upper()}) ---------------------')
    
    # Check for interact_messages.json
    interact_messages_path = os.path.join(process_dir, 'interact_messages.json')
    if not os.path.exists(interact_messages_path):
        print(f"Error: interact_messages.json not found in {process_dir}. Skipping.")
        return {"success": 0, "reasoning": "interact_messages.json not found."}

    try:
        with open(interact_messages_path, 'r', encoding='utf-8') as fr:
            it_messages = json.load(fr)
    except json.JSONDecodeError as e:
        print(f"Error decoding interact_messages.json in {process_dir}: {e}. Skipping.")
        return {"success": 0, "reasoning": f"Error decoding interact_messages.json: {e}"}
    except Exception as e:
        print(f"An unexpected error occurred reading interact_messages.json in {process_dir}: {e}. Skipping.")
        return {"success": 0, "reasoning": f"Unexpected error reading interact_messages.json: {e}"}

    if len(it_messages) < 2:
        print('Not enough messages in interact_messages.json for ' + process_dir + '.')
        return {"success": 0, "reasoning": "Not enough messages to find task/answer."}

    # 1. Extract Task Instruction
    task_content = ""
    found_task = False
    for msg in it_messages:
        if msg.get("role") == "user":
            content_list = msg.get("content")
            if isinstance(content_list, list):
                for item in content_list:
                    if item.get("type") == "text":
                        task_text_block = item.get("text", "")
                        match = re.search(r"Task:(.*?)(?:Please interact with|\nOBSERVATION:|\Z)", task_text_block, re.DOTALL)
                        if match:
                            task_content = match.group(1).strip()
                            found_task = True
                            break
            if found_task:
                break
    
    if not found_task or not task_content:
        print(f"Could not extract task content from interact_messages.json in {process_dir}. Skipping.")
        return {"success": 0, "reasoning": "Could not extract task content."}
    print(f"Extracted Task: {task_content[:100]}...")

    # 2. Extract Agent's Final Answer
    ans_info = it_messages[-1]["content"]
    answer_content = ""
    answer_extracted = False
    pattern_ans = r"Action:\s*(?:stop|ANSWER)\s*(.*)" 
    matches_ans = re.search(pattern_ans, ans_info, re.DOTALL)
    
    if matches_ans:
        answer_content = matches_ans.group(1).strip()
        answer_extracted = True
        print(f"Extracted Answer: {answer_content[:100]}...")
    else:
        print(f"Warning: Could not extract definitive answer content from the last message in {process_dir}. Proceeding with screenshot-only evaluation.")
        answer_content = ""
        answer_extracted = False


    # 3. Retrieve and Filter Screenshots
    res_files_in_dir = os.listdir(process_dir)
    pattern_png = r'screenshot(\d+)\.png'
    png_files_with_indices = []
    for filename in res_files_in_dir:
        match = re.search(pattern_png, filename)
        if match:
            png_files_with_indices.append((filename, int(match.group(1))))
    
    if not png_files_with_indices:
        print(f"Warning: No screenshots found in {process_dir}. Cannot evaluate without images.")
        return {"success": 0, "reasoning": "No screenshots found in directory."}
    
    png_files_with_indices.sort(key=lambda x: x[1])

    selected_img_num = img_num
    if not answer_extracted:
        selected_img_num = 3
        print(f"Using {selected_img_num} screenshots for no-answer evaluation.")

    actual_img_count = min(selected_img_num, len(png_files_with_indices))
    end_files_paths = [os.path.join(process_dir, f_name) for f_name, _ in png_files_with_indices[-actual_img_count:]]

    current_user_prompt_template = ""
    if answer_extracted:
        current_user_prompt_template = USER_PROMPT_WITH_ANSWER
    else:
        current_user_prompt_template = USER_PROMPT_NO_ANSWER

    user_prompt_tmp = current_user_prompt_template.replace('<task>', task_content)
    user_prompt_tmp = user_prompt_tmp.replace('<answer>', answer_content)
    user_prompt_tmp = user_prompt_tmp.replace('<num>', str(len(end_files_paths)))

    eval_response_text = ""
    
    # Call LLM API
    if provider == 'openai':
        whole_content_img = []
        for img_path in end_files_paths:
            try:
                b64_img = encode_image_base64(img_path)
                whole_content_img.append({
                    'type': 'image_url',
                    'image_url': {"url": f"data:image/png;base64,{b64_img}"}
                })
            except FileNotFoundError:
                print(f"Warning: Screenshot file not found: {img_path}. Skipping this image.")
                continue
            except Exception as e:
                print(f"Error processing image {img_path} for OpenAI: {e}. Skipping this image.")
                continue

        if not whole_content_img and not answer_extracted:
            print("Warning: No images available for evaluation and no answer extracted. Proceeding with text-only evaluation (may be insufficient).")
            messages = [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': [{'type': 'text', 'text': user_prompt_tmp}]}
            ]
        else:
            messages = [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': [{'type': 'text', 'text': user_prompt_tmp}] + whole_content_img + [{'type': 'text', 'text': "Your verdict:\n"}]}
            ]
        
        while True:
            try:
                print('Calling OpenAI API to get the auto evaluation......')
                openai_response = client.chat.completions.create(model=api_model, messages=messages, max_tokens=1000, seed=42, temperature=0)
                eval_response_text = openai_response.choices[0].message.content
                break
            except Exception as e:
                print(f"OpenAI API call failed: {e}")
                if 'RateLimitError' in type(e).__name__: time.sleep(10)
                elif 'APIError' in type(e).__name__: time.sleep(15)
                elif 'InvalidRequestError' in type(e).__name__ or (hasattr(e, 'code') and e.code == 'content_filter'):
                    print(f"OpenAI InvalidRequestError/ContentFilter: {e}. Cannot recover.")
                    return {"success": 0, "reasoning": f'OpenAI API Error: {e}'}
                else: time.sleep(10)
        
    elif provider == 'gemini':
        prompt_parts = [SYSTEM_PROMPT, "\n\n---\n\n", user_prompt_tmp]
        for img_path in end_files_paths:
            try:
                img_data = Image.open(img_path)
                prompt_parts.append(img_data)
            except FileNotFoundError:
                print(f"Warning: Screenshot file not found: {img_path}. Skipping this image for Gemini.")
                continue
            except Exception as e:
                print(f"Error loading image {img_path} for Gemini: {e}. Skipping this image.")
                continue
        prompt_parts.append("\nYour verdict:\n")
        
        if not any(isinstance(p, Image.Image) for p in prompt_parts) and not answer_extracted:
             print("Warning: No images available for Gemini evaluation and no answer extracted. Proceeding with text-only evaluation (may be insufficient).")
        
        model = client
        while True:
            try:
                print('Calling Gemini API to get the auto evaluation......')
                gemini_response = model.generate_content(
                    prompt_parts, 
                    generation_config=genai.types.GenerationConfig(temperature=0.0)
                )
                
                if gemini_response.prompt_feedback and gemini_response.prompt_feedback.block_reason.name != "BLOCK_REASON_UNSPECIFIED":
                    block_reason = gemini_response.prompt_feedback.block_reason.name
                    print(f"Gemini API call blocked. Reason: {block_reason}")
                    return {"success": 0, "reasoning": f"Gemini content filter blocked: {block_reason}"}
                
                eval_response_text = gemini_response.text
                break
            except Exception as e:
                print(f"Gemini API call failed: {e}")
                time.sleep(10)

    print('API call complete.')
    print("API Response Content:")
    print(eval_response_text)

    if 'NOT SUCCESS' in eval_response_text:
        is_successful = 0
    elif 'SUCCESS' in eval_response_text:
        is_successful = 1
    else:
        is_successful = 0

    evaluation_result = {
        "reasoning": eval_response_text,
        "success": is_successful,
        "answer_extracted": answer_extracted
    }
    
    print('Auto_eval_res:', evaluation_result["success"])
    print()
        
    return evaluation_result

def main():
    parser = argparse.ArgumentParser(description="Evaluate web tasks using LLMs (OpenAI or Gemini).")
    parser.add_argument('--process_dir', type=str, required=True, help="Root directory containing task folders.")
    parser.add_argument('--provider', type=str, default='gemini', choices=['openai', 'gemini'], help="The provider for the evaluation model.")
    parser.add_argument("--openai_api_key", default=os.getenv("OPENAI_API_KEY"), type=str, help="Your OpenAI API key.")
    parser.add_argument("--gemini_api_key", default=os.getenv("GEMINI_API_KEY"), type=str, help="Your Google Gemini API key.")
    parser.add_argument("--api_model", default="gemini-2.5-pro", type=str, help="API model name.")
    parser.add_argument("--max_attached_imgs", type=int, default=15, help="Number of last screenshots to attach.")
    parser.add_argument('--output_report_path', type=str, required=True, help="Path to save the final evaluation summary report.")
    parser.add_argument('--force-reeval', action='store_true', help="Force re-evaluation of all tasks.")
    parser.add_argument('--mind2web_json', type=str, required=True, help="Path to mind2web_test_cross_web.jsonl (containing domain info).")

    args = parser.parse_args()

    print("Starting evaluation with the following parameters:")
    print(f"  - Process Directory: {args.process_dir}")
    print(f"  - Provider: {args.provider}")
    print(f"  - Model: {args.api_model}")
    print(f"  - Max Attached Images: {args.max_attached_imgs}")
    print(f"  - Force Re-evaluation: {args.force_reeval}")
    print(f"  - Output Report: {args.output_report_path}")
    print(f"  - Mind2Web JSON: {args.mind2web_json}")

    # Initialize Client
    client = None
    if args.provider == 'openai':
        if not args.openai_api_key:
            raise ValueError("Please provide an OpenAI API key.")
        client = OpenAI(api_key=args.openai_api_key)
        print("Using OpenAI for evaluation.")
    elif args.provider == 'gemini':
        if not args.gemini_api_key:
            raise ValueError("Please provide a Gemini API key.")
        try:
            genai.configure(api_key=args.gemini_api_key)
            client = genai.GenerativeModel(args.api_model)
            print("Using Gemini for evaluation.")
        except Exception as e:
            raise RuntimeError(f"Failed to configure Gemini API or load model: {e}")

    # Define all domains
    domains = ['Entertainment', 'Shopping', 'Travel']

    # Load Task ID to Domain Mapping
    mind2web_data_path = os.path.join(os.path.dirname(__file__), args.mind2web_json)
    if not os.path.exists(mind2web_data_path):
        print(f"Error: Data file not found at {mind2web_data_path}. Please ensure it exists.")
        exit(1)

    task_id_to_domain = {} # 存储 task_id -> domain 的映射
    try:
        with open(mind2web_data_path, 'r', encoding='utf-8') as fm:
            for line in fm:
                item = json.loads(line.strip())
                # Extract domain from JSONL
                task_id_to_domain[item['id']] = item['domain']
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading mind2web_test_cross_task.jsonl: {e}")
        exit(1)

    # Initialize/Load Summary Report
    overall_results = {}
    if os.path.exists(args.output_report_path) and not args.force_reeval:
        print(f"Found existing summary report at {args.output_report_path}. Attempting to load.")
        try:
            with open(args.output_report_path, 'r') as f:
                overall_results = json.load(f)
            if not all(k in overall_results for k in ["processed_tasks", "total_tasks", "by_domain"]):
                print("   - WARNING: Existing report format is incomplete or old. Starting fresh.")
                overall_results = {}
            else:
                overall_results["processed_tasks"] = set(overall_results["processed_tasks"])
                print(f"   - Loaded {len(overall_results['processed_tasks'])} previously processed tasks.")
        except (json.JSONDecodeError, IOError) as e:
            print(f"   - WARNING: Could not read or parse existing report. Starting fresh. Error: {e}")
            overall_results = {}
    
    if not overall_results:
        print("Initializing a new summary report.")
        overall_results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": args.api_model,
            "provider": args.provider,
            "processed_tasks": set(),
            "total_tasks": 0,
            "total_success": 0,
            "total_failure": 0,
            "success_rate": 0.0,
            "by_domain": {},
            "failed_tasks_reasoning": []
        }
    else:
        if args.force_reeval:
            print("Force re-evaluation activated. Resetting all aggregate statistics.")
            overall_results.update({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_tasks": 0,
                "total_success": 0,
                "total_failure": 0,
                "success_rate": 0.0,
                "by_domain": {},
                "failed_tasks_reasoning": [],
                "processed_tasks": set()
            })

    # Initialize domain stats
    for domain_name in domains:
        if domain_name not in overall_results["by_domain"]:
            overall_results["by_domain"][domain_name] = {"count": 0, "success_count": 0, "success_rate": 0.0}


    all_task_dirs_in_root = [d for d in os.listdir(args.process_dir) 
                             if os.path.isdir(os.path.join(args.process_dir, d)) and d.startswith('task')]
    all_task_dirs_in_root.sort()

    for task_dir_name in all_task_dirs_in_root:
        original_task_id = task_dir_name[len('task'):]
        
        # Get domain from mapping
        task_domain = task_id_to_domain.get(original_task_id)
        if not task_domain:
            print(f"Warning: Task ID '{original_task_id}' (from folder '{task_dir_name}') not found in domain mapping. Skipping.")
            continue

        file_dir = os.path.join(args.process_dir, task_dir_name)
        
        if original_task_id in overall_results["processed_tasks"] and not args.force_reeval:
            print(f"--> Task {original_task_id} already processed and aggregated. Skipping.")
            continue

        print(f"\n>>> Processing task {original_task_id} in directory {file_dir}")

        eval_file_path = os.path.join(file_dir, f'eval_res_structured_{args.provider}.json')
        evaluation_data = None
        
        if not os.path.exists(eval_file_path) or args.force_reeval:
            print(f"    - Running LLM evaluation for {original_task_id}...")
            evaluation_data = auto_eval(file_dir, client, args.api_model, args.max_attached_imgs, args.provider)
            
            if evaluation_data and "success" in evaluation_data and "reasoning" in evaluation_data:
                try:
                    with open(eval_file_path, 'w', encoding='utf-8') as fw:
                        json.dump(evaluation_data, fw, indent=2, ensure_ascii=False)
                    print(f"    - New individual report saved to '{eval_file_path}'")
                except IOError as e:
                    print(f"    - CRITICAL: Failed to write individual report for {original_task_id}! Error: {e}")
            else:
                print(f"    - LLM evaluation for {original_task_id} returned invalid data. Skipping aggregation.")
                evaluation_data = {"success": 0, "reasoning": "LLM evaluation failed or returned invalid data.", "answer_extracted": False}
                try:
                    with open(eval_file_path, 'w', encoding='utf-8') as fw:
                        json.dump(evaluation_data, fw, indent=2, ensure_ascii=False)
                    print(f"    - Saved default failure report for {original_task_id} to '{eval_file_path}'.")
                except IOError as e:
                    print(f"    - CRITICAL: Failed to write default failure report for {original_task_id}! Error: {e}")

        else:
            print(f"    - Existing individual report found. Loading result from '{eval_file_path}'...")
            try:
                with open(eval_file_path, 'r', encoding='utf-8') as f:
                    evaluation_data = json.load(f)
                if not ("success" in evaluation_data and "reasoning" in evaluation_data):
                    print(f"    - WARNING: Individual report '{eval_file_path}' is incomplete or malformed. Will consider as failed.")
                    evaluation_data = {"success": 0, "reasoning": "Loaded report was incomplete/malformed.", "answer_extracted": False}
                print(f"    - Loaded successfully. Success: {evaluation_data.get('success', 'N/A')}. Skipping API call.")
            except (json.JSONDecodeError, IOError) as e:
                print(f"    - WARNING: Could not read or parse individual report '{eval_file_path}'. Will consider as failed. Error: {e}")
                evaluation_data = {"success": 0, "reasoning": f"Error loading existing report: {e}", "answer_extracted": False}

        if evaluation_data and "success" in evaluation_data:
            task_success_status = evaluation_data['success']

            overall_results["processed_tasks"].add(original_task_id)

            overall_results["total_tasks"] += 1
            if task_success_status == 1:
                overall_results["total_success"] += 1
            else:
                overall_results["total_failure"] += 1
                overall_results["failed_tasks_reasoning"].append({
                    "task_id": original_task_id,
                    "reasoning": evaluation_data.get("reasoning", "No reasoning provided."),
                    "answer_extracted": evaluation_data.get("answer_extracted", False)
                })

            # Update domain stats
            overall_results["by_domain"][task_domain]["count"] += 1
            if task_success_status == 1:
                overall_results["by_domain"][task_domain]["success_count"] += 1

            if overall_results["total_tasks"] > 0:
                overall_results["success_rate"] = overall_results["total_success"] / overall_results["total_tasks"]
            else:
                overall_results["success_rate"] = 0.0

            if overall_results["by_domain"][task_domain]["count"] > 0:
                overall_results["by_domain"][task_domain]["success_rate"] = \
                    overall_results["by_domain"][task_domain]["success_count"] / overall_results["by_domain"][task_domain]["count"]
            else:
                overall_results["by_domain"][task_domain]["success_rate"] = 0.0
            
            try:
                overall_results["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
                overall_results_for_save = overall_results.copy()
                overall_results_for_save["processed_tasks"] = list(overall_results_for_save["processed_tasks"])

                with open(args.output_report_path, 'w', encoding='utf-8') as f:
                    json.dump(overall_results_for_save, f, indent=4, ensure_ascii=False)
                print(f"    - Real-time summary report '{args.output_report_path}' has been updated.")
            except IOError as e:
                print(f"    - WARNING: Failed to write real-time summary! Error: {e}")
        else:
            print(f"    - Invalid evaluation data for {original_task_id}. Skipping aggregation.")
    
    print("\n\n=====================================================")
    print(f"      Final Evaluation Summary Report            ")
    print("=====================================================")
    print(f"Provider: {overall_results.get('provider', 'N/A')}, Model: {overall_results.get('model', 'N/A')}")
    print(f"Report Generated: {overall_results.get('timestamp', 'N/A')}")
    print(f"Total tasks processed: {overall_results['total_tasks']}")
    print(f"Overall Success Rate: {overall_results['success_rate']:.2%}")
    print(f"Total successes: {overall_results['total_success']}")
    print(f"Total failures: {overall_results['total_failure']}")

    print("\n--- Success Rate by Domain ---")
    for domain_name, stats in overall_results["by_domain"].items():
        if stats["count"] > 0:
            print(f"  {domain_name}: {stats['success_count']}/{stats['count']} = {stats['success_rate']:.2%}")
        else:
            print(f"  {domain_name}: No tasks processed.")

    if overall_results['total_failure'] > 0:
        print("\n--- Details of Failed Tasks (first 10, check report file for all) ---")
        for i, failure in enumerate(overall_results.get("failed_tasks_reasoning", [])[:10]):
            print(f"{i+1}. Task ID: {failure['task_id']} (Answer Extracted: {failure.get('answer_extracted', 'N/A')})")
            print(f"   Reasoning: {failure['reasoning'].splitlines()[0]}...")
            
    print(f"\nFull report saved to: {args.output_report_path}")

if __name__ == "__main__":
    main()
