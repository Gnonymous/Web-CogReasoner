import platform
import argparse
import time
import json
import re
import os
import shutil
import logging
import requests
import base64
import copy
import subprocess 

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.service import Service 
from prompts import Qwen_Prompt

from openai import OpenAI
from utils import encode_image, extract_information, print_message,\
    get_webarena_accessibility_tree, get_pdf_retrieval_ans_from_assistant, retry_with_exponential_backoff

def extract_step_summary(response_msg):
    """
    Extracts content between 'Thought:' and 'Action:'.
    This is used for summarization in summarize_and_clip_history.
    """
    thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:)", response_msg, re.DOTALL)
    if thought_match:
        return thought_match.group(1).strip()
    return None

def summarize_and_clip_history(messages, max_attached_imgs=1):
    """
    Summarizes and clips the message history based on the new requirement.
    - The latest user message is kept untouched.
    - Historical assistant messages are summarized (only Thought part).
    - The first-ever user message (containing the task) is clipped but retains the task description.
    - Other historical user messages are clipped, omitting the accessibility tree and optionally the image.
    """
    if len(messages) <= 1:
        return messages

    processed_messages_reversed = []
    history_to_process = messages[1:-1] # Exclude system prompt and latest user message
    latest_user_message = messages[-1] 

    user_msgs_with_img_count = 0
    if isinstance(latest_user_message.get('content'), list):
        if any(item.get('type') == 'image_url' for item in latest_user_message['content']):
            user_msgs_with_img_count += 1

    step_counter = sum(1 for msg in history_to_process if msg['role'] == 'assistant')

    for msg in reversed(history_to_process):
        if msg['role'] == 'assistant':
            # Here we extract only the Thought for summary
            summary_content = extract_step_summary(msg['content']) 
            if summary_content:
                # Format: "Assistant's Thought (Step X): {Thought content}"
                final_content = f"Assistant's Thought (Step {step_counter}): {summary_content}"
            else:
                final_content = f"Assistant's thought (Step {step_counter}): No explicit thought provided." 
            
            processed_messages_reversed.append({'role': 'assistant', 'content': final_content})
            step_counter -= 1
        
        elif msg['role'] == 'user':
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

    final_messages = [messages[0]] # System prompt
    final_messages.extend(reversed(processed_messages_reversed)) # Processed historical messages
    final_messages.append(latest_user_message) # Latest user message

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
    
    # 仅当 args.api_proxy 有效时才添加代理设置
    if args.api_proxy:
        options.add_argument(f'--proxy-server={args.api_proxy}')
        print(f"Using browser proxy: {args.api_proxy}")

    if args.force_device_scale:
        options.add_argument("--force-device-scale-factor=1")

    if args.headless:
        options.add_argument("--headless=new") # 使用新的无头模式
        options.add_argument("--disable-gpu") # 强烈建议添加
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1920,1080") # 设置一个常见且足够大的分辨率
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


# 移除 call_gpt4v_api 和 call_gemini_api，因为我们只用本地模型
# 如果你需要它们，请确保它们被正确地包含在文件中

@retry_with_exponential_backoff(initial_delay=1, exponential_base=1.05, jitter=False, max_retries=10)
def call_localhost_api(args, localhost_api_url, messages, task_dir): 
    # 这里的 `args.api_key` 如果本地模型不需要，可以设置为 None 或一个占位符。
    # 我将其保留，因为 `OpenAI` 客户端实例化时需要它，即使是空字符串也可以。
    # 传入的 `messages` 已经经过 `summarize_and_clip_history` 处理，
    # 包含了 Base64 图片数据或其占位符。
    
    logging.info(f'Calling Localhost API (OpenAI Compatible) at {localhost_api_url}...')
    
    # --- Start: Modification for printing messages ---
    # 创建消息副本，以便在打印时修改图像 URL，而不影响实际发送给 API 的数据
    messages_to_print = copy.deepcopy(messages)
    for msg in messages_to_print:
        if msg['role'] == 'user' and isinstance(msg['content'], list):
            for item in msg['content']:
                if item.get('type') == 'image_url' and 'image_url' in item:
                    # 替换 Base64 数据为占位符
                    item['image_url']['url'] = "<Image data (Base64) - Path unknown, replaced for logging>"
    
    # 打印准备发送给模型的 messages 格式
    logging.info(f"Messages sent to model:\n{json.dumps(messages_to_print, indent=2, ensure_ascii=False)}")
    # --- End: Modification for printing messages ---

    # 实例化 OpenAI 客户端，但基 URL 指向你的本地模型
    local_client = OpenAI(
        api_key=args.api_key, # 对于 localhost 可能不需要或需要一个占位符，但通常需要传递
        base_url=localhost_api_url,
    )

    try:
        # 调用 chat.completions.create 方法
        llm_response = local_client.chat.completions.create(
            model=args.api_model,        # 使用命令行参数指定的本地模型名称
            messages=messages,           # 这是 OpenAI 格式的对话历史
            temperature=args.temperature, # 使用命令行参数的温度
            # top_p=0.95,             # top_p 也是模型参数，可以作为命令行参数传入
            max_tokens=2048,             
        )
        
        prompt_tokens = llm_response.usage.prompt_tokens
        completion_tokens = llm_response.usage.completion_tokens
        response_text = llm_response.choices[0].message.content

        # --- Start: Modification for printing model response ---
        # 打印模型的回复
        logging.info(f"Model response:\n{response_text}")
        # --- End: Modification for printing model response ---

        logging.info(f'Prompt Tokens: {prompt_tokens}; Completion Tokens: {completion_tokens}')

        call_error = False
        # 返回 tokens, error_flag, text_content
        return prompt_tokens, completion_tokens, call_error, response_text

    except Exception as e:
        # 捕获并记录所有异常
        logging.error(f'Error occurred during localhost API call: {type(e).__name__}, Message: {e}', exc_info=True)
        # 标记为调用失败，并返回 None
        return None, None, True, None


def exec_action_click(info, web_ele, driver_task):
    # update, open too much tabs in some webs
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
    # outer_html = web_ele.get_attribute("outerHTML")
    if (ele_tag_name != 'input' and ele_tag_name != 'textarea') or (ele_tag_name == 'input' and ele_type not in ['text', 'search', 'password', 'email', 'tel']):
        warn_obs = f"note: The web element you're trying to type may not be a textbox, and its tag name is <{web_ele.tag_name}>, type is {ele_type}."
    
    try:
        # Not always work to delete
        web_ele.clear()
        # Another way to delete
        if platform.system() == 'Darwin':
            web_ele.send_keys(Keys.COMMAND + "a")
        else:
            web_ele.send_keys(Keys.CONTROL + "a")
        web_ele.send_keys(" ")
        web_ele.send_keys(Keys.BACKSPACE)
    except:
        pass # 静默处理清除失败

    actions = ActionChains(driver_task)
    actions.click(web_ele).perform()
    actions.pause(1)

    try:
        if 'www.cvs.com' not in driver_task.current_url: # 特定网站的优化，可以删除或修改
            driver_task.execute_script("""window.onkeydown = function(e) {if(e.keyCode == 32 && e.target.type != 'text' && e.target.type != 'textarea' && e.target.type != 'search') {e.preventDefault();}};""")
    except:
        pass # 静默处理脚本注入失败

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
        # 获取实际的 WebElement 对象
        web_ele = driver_task.execute_script("return document.elementFromPoint(arguments[0], arguments[1]);", element_box_center[0], element_box_center[1])
        actions = ActionChains(driver_task)
        # 尝试聚焦元素，以便对其进行滚动操作
        driver_task.execute_script("arguments[0].focus();", web_ele)
        if scroll_content == 'down':
            # 尝试使用 Keys.PAGE_DOWN 或发送滚轮事件
            # actions.send_keys_to_element(web_ele, Keys.PAGE_DOWN).perform()
            driver_task.execute_script("arguments[0].scrollTop += arguments[0].offsetHeight;", web_ele) # 元素内部滚动
            # actions.send_keys(Keys.PAGE_DOWN).perform() # 对当前聚焦的元素
        else:
            # actions.send_keys_to_element(web_ele, Keys.PAGE_UP).perform()
            driver_task.execute_script("arguments[0].scrollTop -= arguments[0].offsetHeight;", web_ele) # 元素内部滚动
            # actions.send_keys(Keys.PAGE_UP).perform() # 对当前聚焦的元素
    time.sleep(3)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file', type=str, default='data/test.json')
    parser.add_argument('--max_iter', type=int, default=5)
    parser.add_argument("--api_localhost", default="", type=str, help="localhost API base URL, e.g., 'http://localhost:8000/v1'")
    parser.add_argument("--api_gemini", action='store_true', help="Use Gemini API") # 这个参数现在应该被忽略或用于指示旧的模式
    parser.add_argument("--api_file_assistant", action='store_true', help="Assistant API to analyse files")
    parser.add_argument("--api_key", default="sk-no-key-required", type=str, help="API key for local model (if needed) or cloud APIs. Default is for local.")
    parser.add_argument("--api_model", default="qwen2vl", type=str, help="API model name. E.g., 'qwen2vl' for local, 'gemini-1.5-pro-latest' for Gemini.")
    parser.add_argument("--output_dir", type=str, default='./results')
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max_attached_imgs", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.8) # 默认温度设置为0.1，更倾向于确定性
    parser.add_argument("--download_dir", type=str, default="downloads")
    parser.add_argument("--text_only", action='store_true') # 这个参数可能不再需要
    # for web browser
    parser.add_argument("--headless", action='store_true', help='The window of selenium')
    parser.add_argument("--api_proxy", type=str, default="socks5://127.0.0.1:10808", help="Proxy for API calls and browser (e.g., socks5://127.0.0.1:10808)")
    parser.add_argument("--save_accessibility_tree", action='store_true')
    parser.add_argument("--force_device_scale", action='store_true')
    parser.add_argument("--window_width", type=int, default=1920) # 增大默认窗口大小
    parser.add_argument("--window_height", type=int, default=1080) # 增大默认窗口大小
    parser.add_argument("--fix_box_color", action='store_true') # 这个参数可能不再需要

    args = parser.parse_args()

    # Client setup: Only for localhost, or external APIs if needed
    # 如果 args.api_localhost 非空，则表示使用本地模型
    if args.api_localhost:
        # local_client 会在 call_localhost_api 内部创建
        # 不需要在这里显式创建 client 变量
        print(f"Configured to use Localhost API at: {args.api_localhost}")
    elif args.api_gemini: # 保留以防万一，但不再是主要关注点
        print("WARNING: api_gemini is set. This script is optimized for localhost API. Please ensure your setup supports Gemini.")
        import google.generativeai as genai # 仅在需要时导入
        genai.configure(api_key=args.api_key)
        client = genai.GenerativeModel(args.api_model)
    else: # 默认情况，可能是不使用任何特定API，这在纯本地模式下不常见
        print("WARNING: Neither api_localhost nor api_gemini is set. Assuming no external API will be called.")
        client = None # 或根据你的实际情况初始化为OpenAI客户端，如果仍有OpenAI调用

    # 文件助手 API 客户端（如果需要，且通常是OpenAI的API）
    client_file_assis = None
    if args.api_file_assistant:
        if args.api_key == "sk-no-key-required": # 如果是本地模式默认key，则提醒
            print("WARNING: Assistant API requires a valid OpenAI API key. Using default 'sk-no-key-required'.")
        client_file_assis = OpenAI(api_key=args.api_key)
        print("Using OpenAI Assistant API for file analysis.")


    options = driver_config(args)
    print("WebDriver configuration complete.")
    
    result_dir = args.output_dir
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(args.download_dir, exist_ok=True) # 确保下载目录存在

    tasks = []
    try:
        with open(args.test_file, 'r', encoding='utf-8') as f:
            for line in f:
                tasks.append(json.loads(line))
        print (f'Loaded {len(tasks)} tasks from {args.test_file}')
    except json.JSONDecodeError as e:
        print(f"Error reading JSON from {args.test_file}: {e}")
        print("Please ensure your test file is a valid JSON Lines (.jsonl) file.")
        return # 退出程序

    for task_obj in tasks: 
        force_cleanup_processes() # 每次任务前清理进程

        task_id_str = task_obj["id"]
        task_dir = os.path.join(result_dir, 'task{}'.format(task_id_str))
        
        # 检查任务是否已完成（通过检查 interact_messages.json）
        if os.path.exists(os.path.join(task_dir, 'interact_messages.json')):
            print (f'This task {task_id_str} has been processed. Skipping.')
            continue
        
        os.makedirs(task_dir, exist_ok=True)
        setup_logger(task_dir)
        logging.info(f'########## TASK{task_id_str} ##########')
        print(f'########## TASK{task_id_str} ##########')
        
        print(f"--- Attempting to start WebDriver for task {task_id_str}... ---")
        driver_task = None # 初始化为 None
        try:
            driver_task = webdriver.Chrome(options=options)
            print(f"--- WebDriver started successfully! ---")
        except Exception as e:
            print(f"--- ERROR: Failed to start WebDriver: {e} ---")
            logging.error(f"Failed to start WebDriver: {e}", exc_info=True)
            if driver_task: driver_task.quit() # 如果创建了一部分，尝试退出
            continue # 跳过当前任务，尝试下一个

        driver_task.set_window_size(args.window_width, args.window_height)
        print(f"--- Window size set. Attempting to get URL: {task_obj['web']} ---")
        try:
            driver_task.get(task_obj['web'])
            print("--- URL loaded successfully! ---")
            time.sleep(5)
        except Exception as e:
            print(f"--- ERROR: Failed to load URL {task_obj['web']}: {e} ---")
            logging.error(f"Failed to load URL {task_obj['web']}: {e}", exc_info=True)
            if driver_task: driver_task.quit()
            continue # 跳过当前任务，尝试下一个

        try:
            body_element = driver_task.find_element(By.TAG_NAME, 'body')
            body_element.click()
            driver_task.execute_script("""window.onkeydown = function(e) {if(e.keyCode == 32 && e.target.type != 'text' && e.target.type != 'textarea') {e.preventDefault();}};""")
        except Exception as e:
            logging.warning(f"Could not click body or set keydown listener: {e}")

        # Cleanup download dir
        for filename in os.listdir(args.download_dir):
            file_path = os.path.join(args.download_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        download_files = [] 

        fail_obs, pdf_obs, warn_obs = "", "", "" 
        messages = [{'role': 'system', 'content': Qwen_Prompt}] # 使用 Qwen_Prompt 作为系统提示
        
        print(f"Task: {task_obj['ques']}")
        it = 0
        accumulate_prompt_token, accumulate_completion_token = 0, 0
        
        # New variable to store the previous assistant's Thought and Action
        prev_assistant_response_summary = ""

        while it < args.max_iter:
            logging.info(f'Iter: {it+1}')
            print(f"Step: {it+1}")
            it += 1
            
            try:
                accessibility_tree_path = os.path.join(task_dir, 'accessibility_tree{}'.format(it))
                ac_tree, obs_info = get_webarena_accessibility_tree(driver_task, accessibility_tree_path)
            except Exception as e:
                logging.error(f'Driver error when obtaining accessibility tree: {e}', exc_info=True)
                fail_obs = "Failed to get accessibility tree. WebDriver might be in a bad state."
                break 

            img_path = os.path.join(task_dir, 'screenshot{}.png'.format(it))
            driver_task.save_screenshot(img_path)
            b64_img = encode_image(img_path)

            # --- 构造当前用户消息 ---
            obs_prefix_parts = []
            if fail_obs:
                obs_prefix_parts.append(f"Observation Error: {fail_obs}")
            if pdf_obs:
                obs_prefix_parts.append(f"PDF Content: {pdf_obs}")
            if warn_obs:
                obs_prefix_parts.append(f"Warning: {warn_obs}")
            
            # Incorporate previous assistant's response summary if available
            if prev_assistant_response_summary:
                obs_prefix_parts.append(f"Previous Assistant's action result: {prev_assistant_response_summary}")

            obs_prefix_str = " ".join(obs_prefix_parts).strip()
            
            if it == 1:
                text_content = f"Task: {task_obj['ques']}\nOBSERVATION:{' ' + obs_prefix_str if obs_prefix_str else ''}\n<image>\n{ac_tree}"
            else:
                text_content = f"OBSERVATION:{' ' + obs_prefix_str if obs_prefix_str else ''}\n<image>\n{ac_tree}"

            curr_msg = {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': text_content},
                    {'type': 'image_url', 'image_url': {"url": f"data:image/png;base64,{b64_img}"}}
                ]
            }
            messages.append(curr_msg)
            # --- 构造当前用户消息结束 ---

            # Prepare messages for the API call using the summarization function
            messages_for_api = summarize_and_clip_history(copy.deepcopy(messages), args.max_attached_imgs)
            
            prompt_tokens, completion_tokens, call_error, model_raw_response = None, None, False, None

            if args.api_localhost:
                prompt_tokens, completion_tokens, call_error, model_raw_response = call_localhost_api(args, args.api_localhost, messages_for_api, task_dir)
            elif args.api_gemini:
                # Assuming 'client' is initialized for Gemini if this branch is taken
                prompt_tokens, completion_tokens, call_error, model_raw_response = call_gemini_api(args, messages_for_api, task_dir)
            else: # OpenAI
                # Assuming 'client' is initialized for OpenAI if this branch is taken
                prompt_tokens, completion_tokens, call_error, openai_api_response_obj = call_gpt4v_api(args, client, messages_for_api)
                if not call_error:
                    model_raw_response = openai_api_response_obj.choices[0].message.content


            if call_error or model_raw_response is None:
                logging.error(f"API call failed or returned empty response for task {task_id_str}. Breaking loop.")
                break 
            
            accumulate_prompt_token += prompt_tokens if prompt_tokens is not None else 0
            accumulate_completion_token += completion_tokens if completion_tokens is not None else 0
            logging.info(f'Accumulate Prompt Tokens: {accumulate_prompt_token}; Accumulate Completion Tokens: {accumulate_completion_token}')
            
            messages.append({'role': 'assistant', 'content': model_raw_response})

            # Reset observation variables for the next loop
            fail_obs, pdf_obs, warn_obs = "", "", ""
            
            # --- START: Action/Thought extraction and execution (adapted for new format) ---
            bot_thought = ""
            chosen_action = ""

            # 匹配 "Thought:" 和 "Action:" 结构
            thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:)", model_raw_response, re.DOTALL)
            action_match = re.search(r"Action:\s*(.*)", model_raw_response, re.DOTALL)

            if thought_match:
                bot_thought = thought_match.group(1).strip()
            else:
                logging.warning(f"Model response missing 'Thought:' pattern. Response: {model_raw_response}")
                bot_thought = "Thought: No explicit thought provided or format mismatch."

            if action_match:
                chosen_action = action_match.group(1).strip()
            else:
                fail_obs = "Format ERROR: Model response missing 'Action:' pattern."
                logging.error(f"Model response parsing failed: {fail_obs}\nResponse: {model_raw_response}")
                continue 

            print(f'Thought: {bot_thought}')
            print(f'Action: {chosen_action}')

            # Store previous assistant's response for the next turn's observation
            # This is the core modification for "previous assistant's reply"
            prev_assistant_response_summary = f"Thought: {bot_thought}\nAction: {chosen_action}"


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
                    
                    if args.api_file_assistant:
                        current_files = sorted(os.listdir(args.download_dir))
                        if current_files != download_files:
                            time.sleep(10) 
                            current_files = sorted(os.listdir(args.download_dir))
                            current_download_file = [pdf for pdf in current_files if pdf not in download_files and pdf.endswith('.pdf')]
                            if current_download_file:
                                pdf_file = current_download_file[0]
                                pdf_obs = get_pdf_retrieval_ans_from_assistant(client_file_assis, os.path.join(args.download_dir, pdf_file), task_obj['ques'])
                                shutil.copy(os.path.join(args.download_dir, pdf_file), task_dir)
                                pdf_obs = "You downloaded a PDF file, I ask the Assistant API to answer the task based on the PDF file and get the following response: " + pdf_obs
                            download_files = current_files 

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
                    if 'wolfram' in task_obj['web']:
                        time.sleep(5)

                elif action_key == 'scroll':
                    exec_action_scroll(info, None, driver_task, args, obs_info)

                elif action_key == 'go_back':
                    driver_task.back()
                    time.sleep(2)
                
                elif action_key == 'restart':
                    logging.info(f"Executing RESTART action, returning to initial URL: {task_obj['web']}")
                    driver_task.get(task_obj['web'])
                    time.sleep(5)

                elif action_key in ['answer', 'stop']:
                    final_answer = info['content']
                    logging.info(f"Final Answer (from '{action_key}' action): {final_answer}")
                    logging.info('Task finished successfully!!')
                    break 
                else:
                    raise NotImplementedError(f"Unsupported action key: {action_key}")
            except Exception as e:
                logging.error(f'Driver execution error: {e}', exc_info=True)
                if 'element click intercepted' not in str(e) and 'element not interactable' not in str(e): 
                    fail_obs = "The action you have chosen cannot be executed. Please double-check if you have selected the correct element or used correct action format."
                else:
                    fail_obs = "" 
                time.sleep(2)
            # --- END: Action/Thought extraction and execution ---
        
        print_message(messages, task_dir, is_v=True)
        
        if driver_task: 
            driver_task.quit()
        
        logging.info(f'Total tokens for task {task_id_str}: Prompt: {accumulate_prompt_token}, Completion: {accumulate_completion_token}')


if __name__ == '__main__':
    main()
