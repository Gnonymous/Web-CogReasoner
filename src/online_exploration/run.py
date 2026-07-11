import platform
import argparse
import time
import json
import re
import os
import logging
import copy

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from prompts import CogReasoner_Prompt
from openai import OpenAI
from utils import encode_image, extract_information, print_message, get_webarena_accessibility_tree, retry_with_exponential_backoff


def extract_step_summary(response_msg):
    """
    Extract the model's step summary block.
    """
    summary_match = re.search(
        r"### Step Summary:\n(.*?)(?=\n### Final Action.*?(?=\n|\Z)|\Z)", 
        response_msg, 
        re.DOTALL
    )
    if summary_match:
        return summary_match.group(1).strip()
    return None

def summarize_and_clip_history(messages, max_attached_imgs=1):
    """
    Summarize and clip the message history to save tokens.
    """
    if len(messages) <= 1:
        return messages

    processed_messages_reversed = []
    history = messages[1:]
    user_msgs_with_img_count = 0
    is_first_user_msg_processed = False
    step_counter = sum(1 for msg in history if msg['role'] == 'assistant')

    for msg in reversed(history):
        if msg['role'] == 'assistant':
            summary_content = extract_step_summary(msg['content'])
            if summary_content:
                final_content = f"Step {step_counter} Summary: {summary_content}"
            else:
                final_content = f"Step {step_counter} Summary: Action was taken."
            
            processed_messages_reversed.append({'role': 'assistant', 'content': final_content})
            step_counter -= 1
        
        elif msg['role'] == 'user':
            if not is_first_user_msg_processed:
                processed_messages_reversed.append(msg)
                is_first_user_msg_processed = True
                user_msgs_with_img_count += 1
            else:
                should_attach_image = (user_msgs_with_img_count < max_attached_imgs)
                
                original_text = ""
                if isinstance(msg.get('content'), list):
                    original_text = next((item['text'] for item in msg['content'] if item['type'] == 'text'), "")

                is_task_message = original_text.lstrip().startswith("Task:")
                
                new_text = ""
                if is_task_message:
                    task_line = original_text.split('\n', 1)[0]
                    if should_attach_image:
                        new_text = f"{task_line}\nOBSERVATION:\n<image> (accessibility tree omitted)."
                    else:
                        new_text = f"{task_line}\nOBSERVATION:\n(image and accessibility tree omitted)."
                else:
                    if should_attach_image:
                        new_text = "OBSERVATION:\n<image> (accessibility tree omitted)."
                    else:
                        new_text = "OBSERVATION:\n(image and accessibility tree omitted)."

                new_content = [{'type': 'text', 'text': new_text}]

                if should_attach_image:
                    original_image = next((item for item in msg['content'] if item['type'] == 'image_url'), None)
                    if original_image:
                        new_content.append(original_image)
                    user_msgs_with_img_count += 1
                
                processed_messages_reversed.append({'role': 'user', 'content': new_content})

    final_messages = [messages[0]]
    final_messages.extend(reversed(processed_messages_reversed))

    return final_messages

def setup_logger(folder_path):
    log_file_path = os.path.join(folder_path, 'agent.log')

    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    handler = logging.FileHandler(log_file_path)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

def force_cleanup_processes():
    """
    Avoid terminating unrelated browser sessions on shared hosts.
    """
    print("Skipping global Chrome/IPC cleanup; WebDriver sessions are closed individually.")

def driver_config(args):

    options = webdriver.ChromeOptions()

    if args.save_accessibility_tree:
        args.force_device_scale = True
    if args.api_proxy:
        options.add_argument(f'--proxy-server={args.api_proxy}')

    if args.force_device_scale:
        options.add_argument("--force-device-scale-factor=1")

    if args.headless:
        options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1920,1080")
        options.add_argument(
            "user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        )
    options.add_experimental_option(
        "prefs", {
            "download.default_directory": args.download_dir,
            "plugins.always_open_pdf_externally": True
        }
    )

    return options

@retry_with_exponential_backoff(initial_delay=1, exponential_base=1.05, jitter=False, max_retries=10)
def call_localhost_api(args, localhost_api_url, messages, task_dir): 
    retry_times = 0
    while True:
        try:
            logging.info(f'Calling Localhost API (OpenAI Compatible) at {localhost_api_url}...')
            messages_to_print = copy.deepcopy(messages)
            for msg in messages_to_print:
                if msg['role'] == 'user' and isinstance(msg['content'], list):
                    for item in msg['content']:
                        if item.get('type') == 'image_url' and 'image_url' in item:
                            item['image_url']['url'] = "<Image data (Base64) - Path unknown, replaced for logging>"
            
            logging.info(f"Messages sent to model:\n{json.dumps(messages_to_print, indent=2, ensure_ascii=False)}")

            local_client = OpenAI(
                api_key=args.api_key,
                base_url=localhost_api_url,
            )

            llm_response = local_client.chat.completions.create(
                model=args.model_name,
                messages=messages,
                temperature=0.1, 
                top_p=0.95,             
                max_tokens=2048,             
            )
            
            prompt_tokens = llm_response.usage.prompt_tokens
            completion_tokens = llm_response.usage.completion_tokens
            response_text = llm_response.choices[0].message.content

            logging.info(f"Model response:\n{response_text}")
            logging.info(f'Prompt Tokens: {prompt_tokens}; Completion Tokens: {completion_tokens}')

            call_error = False
            return prompt_tokens, completion_tokens, call_error, response_text

        except Exception as e:
            logging.info(f'Error occurred, retrying. Error type: {type(e).__name__}, Message: {e}')

            if type(e).__name__ == 'RateLimitError':
                time.sleep(10)
            elif type(e).__name__ == 'APIError':
                time.sleep(15)
            elif type(e).__name__ == 'InvalidRequestError':
                call_error = True
                return None, None, call_error, None
            else:
                call_error = True
                return None, None, call_error, None

        retry_times += 1
        if retry_times == 10:
            logging.info('Retrying too many times for localhost API call.')
            return None, None, True, None


def exec_action_click(info, web_ele, driver_task):
    original_tabs = driver_task.window_handles
    driver_task.execute_script("arguments[0].setAttribute('target', '_self')", web_ele)
    web_ele.click()
    time.sleep(3)
    new_tabs = driver_task.window_handles
    if len(new_tabs) > len(original_tabs):
        new_tab = [tab for tab in new_tabs if tab not in original_tabs][0]
        driver_task.switch_to.window(new_tab)
        new_tab_url = driver_task.current_url
        driver_task.close()
        driver_task.switch_to.window(original_tabs[0])
        driver_task.get(new_tab_url)
        time.sleep(2)


def exec_action_type(info, web_ele, driver_task):
    warn_obs = ""
    type_content = info['content']

    ele_tag_name = web_ele.tag_name.lower()
    ele_type = web_ele.get_attribute("type")
    if (ele_tag_name != 'input' and ele_tag_name != 'textarea') or (ele_tag_name == 'input' and ele_type not in ['text', 'search', 'password', 'email', 'tel']):
        warn_obs = f"note: The web element you're trying to type may not be a textbox, and its tag name is <{web_ele.tag_name}>, type is {ele_type}."
    
    try:
        web_ele.clear()
        if platform.system() == 'Darwin':
            web_ele.send_keys(Keys.COMMAND + "a")
        else:
            web_ele.send_keys(Keys.CONTROL + "a")
        web_ele.send_keys(" ")
        web_ele.send_keys(Keys.BACKSPACE)
    except:
        pass

    actions = ActionChains(driver_task)
    actions.click(web_ele).perform()
    actions.pause(1)

    try:
        if 'www.cvs.com' not in driver_task.current_url:
            driver_task.execute_script("""window.onkeydown = function(e) {if(e.keyCode == 32 && e.target.type != 'text' && e.target.type != 'textarea' && e.target.type != 'search') {e.preventDefault();}};""")
    except:
        pass

    actions.send_keys(type_content)
    actions.pause(2)

    actions.send_keys(Keys.ENTER)
    actions.perform()
    time.sleep(10)
    return warn_obs

def calculate_intersection(args, bound_box):
    x1, y1, w1, h1 = (0, 0, args.window_width, args.window_height)
    x2, y2, w2, h2 = bound_box

    left = max(x1, x2)
    top = max(y1, y2)
    right = min(x1 + w1, x2 + w2)
    bottom = min(y1 + h1, y2 + h2)

    width = max(0, right - left)
    height = max(0, bottom - top)

    return [left, top, width, height]

def exec_action_scroll(info, web_eles, driver_task, args, obs_info):
    scroll_ele_number = info['number']
    scroll_content = info['content']
    if scroll_ele_number == "WINDOW":
        if scroll_content == 'down':
            driver_task.execute_script(f"window.scrollBy(0, {args.window_height*2//3});")
        else:
            driver_task.execute_script(f"window.scrollBy(0, {-args.window_height*2//3});")
    else:
        
        scroll_ele_number = int(scroll_ele_number)
        element_box = obs_info[scroll_ele_number]['union_bound']
        element_box = calculate_intersection(args, element_box)
        element_box_center = (element_box[0] + element_box[2] // 2, element_box[1] + element_box[3] // 2)
        web_ele = driver_task.execute_script("return document.elementFromPoint(arguments[0], arguments[1]);", element_box_center[0], element_box_center[1])
        actions = ActionChains(driver_task)
        driver_task.execute_script("arguments[0].focus();", web_ele)
        if scroll_content == 'down':
            actions.key_down(Keys.ALT).send_keys(Keys.ARROW_DOWN).key_up(Keys.ALT).perform()
        else:
            actions.key_down(Keys.ALT).send_keys(Keys.ARROW_UP).key_up(Keys.ALT).perform()
    time.sleep(3)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file', type=str, default='data/test.json')
    parser.add_argument('--model_name', type=str, default='qwen2vl', help='Name of the model to use for inference')
    parser.add_argument('--max_iter', type=int, default=5)
    parser.add_argument("--api_localhost", default="", type=str, help="localhost API")
    parser.add_argument("--api_key", type=str, default="", help="Localhost API key (if required by the server)")
    parser.add_argument("--output_dir", type=str, default='./results')
    parser.add_argument("--auto_quit", action='store_true', help='Quit WebDriver at the end of each task')
    parser.add_argument("--max_attached_imgs", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--download_dir", type=str, default="downloads")
    parser.add_argument("--headless", action='store_true', help='Run Chrome in headless mode')
    parser.add_argument("--api_proxy", type=str, default="", help="Proxy for API calls (e.g., socks5://127.0.0.1:10808)")
    parser.add_argument("--save_accessibility_tree", action='store_true')
    parser.add_argument("--force_device_scale", action='store_true')
    parser.add_argument("--window_width", type=int, default=1024)
    parser.add_argument("--window_height", type=int, default=1024)
    parser.add_argument("--fix_box_color", action='store_true')

    args = parser.parse_args()
    os.makedirs(args.download_dir, exist_ok=True)

    if not args.api_localhost:
        raise ValueError("--api_localhost is required for local model usage.")
    client_localhost = args.api_localhost

    options = driver_config(args)
    print("config_finish")
    
    result_dir = args.output_dir
    os.makedirs(result_dir, exist_ok=True)

    tasks = []
    with open(args.test_file, 'r', encoding='utf-8') as f:
        for line in f:
            tasks.append(json.loads(line))
    print ('number of tasks', len(tasks))

    for task_id in range(len(tasks)):
        if args.auto_quit:
            force_cleanup_processes()

        task = tasks[task_id]
        task_dir = os.path.join(result_dir, 'task{}'.format(task["id"]))
        if os.path.exists(os.path.join(task_dir, 'interact_messages.json')):
            print ('This task has been processed', task["id"])
            continue
        os.makedirs(task_dir, exist_ok=True)
        setup_logger(task_dir)
        logging.info(f'########## TASK{task["id"]} ##########')
        print(f'########## TASK{task["id"]} ##########')
        
        print(f"--- Attempting to start WebDriver for task {task['id']}... ---")
        driver_task = webdriver.Chrome(options=options)
        print(f"--- WebDriver started successfully! ---")
        driver_task.set_window_size(args.window_width, args.window_height)
        print(f"--- Window size set. Attempting to get URL: {task['web']} ---")
        driver_task.get(task['web'])
        print("--- URL loaded successfully! ---")
        time.sleep(5)

        try:
            driver_task.find_element(By.TAG_NAME, 'body').click()
            driver_task.execute_script("""window.onkeydown = function(e) {if(e.keyCode == 32 && e.target.type != 'text' && e.target.type != 'textarea') {e.preventDefault();}};""")
        except Exception as e:
            logging.warning(f"Could not click body or set keydown listener: {e}")

        for filename in os.listdir(args.download_dir):
            file_path = os.path.join(args.download_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        fail_obs, warn_obs = "", ""
        messages = [{'role': 'system', 'content': CogReasoner_Prompt}]
        
        print(f"Task: {task['ques']}")
        it = 0
        accumulate_prompt_token, accumulate_completion_token = 0, 0
        
        while it < args.max_iter:
            logging.info(f'Iter: {it+1}')
            print(f"Step: {it+1}")
            it += 1
            
            try:
                accessibility_tree_path = os.path.join(task_dir, 'accessibility_tree{}'.format(it))
                ac_tree, obs_info = get_webarena_accessibility_tree(driver_task, accessibility_tree_path)
            except Exception as e:
                logging.error(f'Driver error when obtaining accessibility tree: {e}')
                break

            img_path = os.path.join(task_dir, 'screenshot{}.png'.format(it))
            driver_task.save_screenshot(img_path)
            b64_img = encode_image(img_path)

            obs_prefix = ""
            if fail_obs:
                obs_prefix = f" {fail_obs}"
            elif warn_obs:
                obs_prefix = f" {warn_obs}"

            if it == 1:
                text_content = f"Task: {task['ques']}\nOBSERVATION:{obs_prefix}\n<image>\n{ac_tree}"
            else:
                text_content = f"OBSERVATION:{obs_prefix}\n<image>\n{ac_tree}"

            curr_msg = {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': text_content},
                    {'type': 'image_url', 'image_url': {"url": f"data:image/png;base64,{b64_img}"}}
                ]
            }
            messages.append(curr_msg)
            messages_ = summarize_and_clip_history(copy.deepcopy(messages), args.max_attached_imgs)
            prompt_tokens, completion_tokens, call_error, model_response = call_localhost_api(
                args, client_localhost, messages_, task_dir
            )

            if call_error:
                break
            
            accumulate_prompt_token += prompt_tokens if prompt_tokens is not None else 0
            accumulate_completion_token += completion_tokens if completion_tokens is not None else 0
            logging.info(f'Accumulate Prompt Tokens: {accumulate_prompt_token}; Accumulate Completion Tokens: {accumulate_completion_token}')
            logging.info('API call complete...')
            
            response_msg = model_response
            messages.append({'role': 'assistant', 'content': model_response})

            fail_obs, warn_obs = "", ""

            bot_thought = ""
            chosen_action = ""

            action_match = re.search(r"### Final Action.*?\s*Action:\s*(.*)", response_msg, re.DOTALL)
            if action_match:
                chosen_action = action_match.group(1).strip()
            else:
                fail_obs = "Format ERROR: Model response missing '### Final Action: Action:' pattern."
                logging.error(f"Model response parsing failed: {fail_obs}\nResponse: {response_msg}")
                continue

            thought_parts = []
            thought_headers = [
                "### Task Clarification", "### Webpage Structure", "### Key Element Analysis",
                "### Trajectory History Review", "### Task Decomposition", 
                "### Step-by-Step Reasoning", "### Step Summary"
            ]
            for header in thought_headers:
                match = re.search(re.escape(header) + r":\n(.*?)(?=\n###|\Z)", response_msg, re.DOTALL)
                if match:
                    thought_parts.append(f"{header}:\n" + match.group(1).strip())
            
            if thought_parts:
                bot_thought = "\n\n".join(thought_parts)
            else:
                bot_thought = "Thought: No explicit detailed thought sections found."
                logging.warning("No explicit detailed thought sections found in model response.")

            print(f'Thought: {bot_thought}')
            print(f'Action: {chosen_action}')

            try:
                action_key, info = extract_information(chosen_action)
            except ValueError as e:
                logging.error(f"Error parsing chosen_action: {e}")
                fail_obs = f"Action format ERROR: {e}. Please use a valid format."
                continue

            try:
                window_handle_task = driver_task.current_window_handle
                driver_task.switch_to.window(window_handle_task)

                if action_key == 'click':
                    click_ele_number = int(info['number'])
                    element_box = obs_info[click_ele_number]['union_bound']
                    element_box = calculate_intersection(args, element_box)
                    element_box_center = (element_box[0] + element_box[2] // 2, element_box[1] + element_box[3] // 2)
                    web_ele = driver_task.execute_script("return document.elementFromPoint(arguments[0], arguments[1]);", element_box_center[0], element_box_center[1])
                    web_ele_attr_info = f"Web Element Info; <role: '{obs_info[click_ele_number]['role']['value']}'; name: '{obs_info[click_ele_number]['name']['value']}'>" 
                    print(web_ele_attr_info)
                    
                    ele_tag_name = web_ele.tag_name.lower()
                    ele_type = web_ele.get_attribute("type")
                    exec_action_click(info, web_ele, driver_task)
                    
                    if ele_tag_name == 'button' and ele_type == 'submit':
                        time.sleep(10)

                elif action_key == 'wait':
                    time.sleep(5)

                elif action_key == 'type':
                    type_ele_number = int(info['number'])
                    element_box = obs_info[type_ele_number]['union_bound']
                    element_box = calculate_intersection(args, element_box)
                    element_box_center = (element_box[0] + element_box[2] // 2, element_box[1] + element_box[3] // 2)
                    web_ele = driver_task.execute_script("return document.elementFromPoint(arguments[0], arguments[1]);", element_box_center[0], element_box_center[1])
                    web_ele_attr_info = f"Web Element Info; <role: '{obs_info[type_ele_number]['role']['value']}'; name: '{obs_info[type_ele_number]['name']['value']}'>" 
                    print(web_ele_attr_info)
                    warn_obs = exec_action_type(info, web_ele, driver_task)
                    if 'wolfram' in task['web']:
                        time.sleep(5)

                elif action_key == 'scroll':
                    exec_action_scroll(info, None, driver_task, args, obs_info)

                elif action_key == 'go_back':
                    driver_task.back()
                    time.sleep(2)
                
                elif action_key == 'restart':
                    logging.info(f"Executing RESTART action, returning to initial URL: {task['web']}")
                    driver_task.get(task['web'])
                    time.sleep(5)

                elif action_key in ['answer', 'stop']:
                    final_answer = info['content']
                    logging.info(f"Final Answer (from '{action_key}' action): {final_answer}")
                    logging.info('Task finished successfully!!')
                    break
                else:
                    raise NotImplementedError
            except Exception as e:
                logging.error('Driver error info:', exc_info=True)
                if 'element click intercepted' not in str(e):
                    fail_obs = "The action you have chosen cannot be executed. Please double-check if you have selected the correct element or used correct action format."
                else:
                    fail_obs = ""
                time.sleep(2)
        print_message(messages, task_dir, is_v=True)
        if args.auto_quit:
            driver_task.quit()



if __name__ == '__main__':
    main()
