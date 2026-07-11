import base64
import re
import os
import json
import time
import logging
import random
from typing import Any, TypedDict

def retry_with_exponential_backoff(
        initial_delay: float = 1,
        exponential_base: float = 1.1,
        jitter: bool = True,
        max_retries: int = 40,
):
    """Retry a function with exponential backoff."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            rate_limit_retry_num = 0
            delay = initial_delay

            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"Try count: {rate_limit_retry_num}, Error: {e}")
                    rate_limit_retry_num += 1

                    if rate_limit_retry_num > max_retries:
                        raise Exception(
                            f"Maximum number of retries ({max_retries}) exceeded."
                        )

                    delay *= exponential_base * (1 + jitter * random.random())
                    print(f"Failure, sleep {delay} secs")
                    time.sleep(delay)
        return wrapper
    return decorator


class AccessibilityTreeNode(TypedDict):
    nodeId: str
    ignored: bool
    role: dict[str, Any]
    chromeRole: dict[str, Any]
    name: dict[str, Any]
    properties: list[dict[str, Any]]
    childIds: list[str]
    parentId: str
    backendDOMNodeId: str
    frameId: str
    bound: list[float] | None
    union_bound: list[float] | None
    offsetrect_bound: list[float] | None


class BrowserConfig(TypedDict):
    win_top_bound: float
    win_left_bound: float
    win_width: float
    win_height: float
    win_right_bound: float
    win_lower_bound: float
    device_pixel_ratio: float


class BrowserInfo(TypedDict):
    DOMTree: dict[str, Any]
    config: BrowserConfig


IGNORED_ACTREE_PROPERTIES = (
    "focusable",
    "editable",
    "readonly",
    "level",
    "settable",
    "multiline",
    "invalid",
    "hiddenRoot",
    "hidden",
    "controls",
    "labelledby",
    "describedby",
    "url"
)

AccessibilityTree = list[AccessibilityTreeNode]

IN_VIEWPORT_RATIO_THRESHOLD = 0.8


def fetch_browser_info(browser) -> BrowserInfo:
    tree = browser.execute_cdp_cmd(
        "DOMSnapshot.captureSnapshot",
        {
            "computedStyles": [],
            "includeDOMRects": True,
            "includePaintOrder": True,
        },
    )

    bounds = tree["documents"][0]["layout"]["bounds"]
    b = bounds[0]
    n = b[2] / browser.get_window_size()["width"]
    bounds = [[x / n for x in bound] for bound in bounds]
    tree["documents"][0]["layout"]["bounds"] = bounds

    win_top_bound = browser.execute_script("return window.pageYOffset;")
    win_left_bound = browser.execute_script("return window.pageXOffset;")
    win_width = browser.execute_script("return window.innerWidth;")
    win_height = browser.execute_script("return window.innerHeight;")
    win_right_bound = win_left_bound + win_width
    win_lower_bound = win_top_bound + win_height
    device_pixel_ratio = browser.execute_script("return window.devicePixelRatio;")
    assert device_pixel_ratio == 1.0, "devicePixelRatio is not 1.0"

    config: BrowserConfig = {
        "win_top_bound": win_top_bound,
        "win_left_bound": win_left_bound,
        "win_width": win_width,
        "win_height": win_height,
        "win_right_bound": win_right_bound,
        "win_lower_bound": win_lower_bound,
        "device_pixel_ratio": device_pixel_ratio,
    }
    info: BrowserInfo = {"DOMTree": tree, "config": config}

    return info


def get_element_in_viewport_ratio(
    elem_left_bound: float,
    elem_top_bound: float,
    width: float,
    height: float,
    config: BrowserConfig,
) -> float:
    elem_right_bound = elem_left_bound + width
    elem_lower_bound = elem_top_bound + height

    win_left_bound = 0
    win_right_bound = config["win_width"]
    win_top_bound = 0
    win_lower_bound = config["win_height"]

    overlap_width = max(
        0,
        min(elem_right_bound, win_right_bound)
        - max(elem_left_bound, win_left_bound),
    )
    overlap_height = max(
        0,
        min(elem_lower_bound, win_lower_bound)
        - max(elem_top_bound, win_top_bound),
    )

    ratio = overlap_width * overlap_height / width * height
    return ratio


def get_bounding_client_rect(
    browser, backend_node_id: str
) -> dict[str, Any]:
    try:
        remote_object = browser.execute_cdp_cmd(
            "DOM.resolveNode", {"backendNodeId": int(backend_node_id)}
        )
        remote_object_id = remote_object["object"]["objectId"]
        response = browser.execute_cdp_cmd(
            "Runtime.callFunctionOn",
            {
                "objectId": remote_object_id,
                "functionDeclaration": """
                    function() {
                        if (this.nodeType == 3) {
                            var range = document.createRange();
                            range.selectNode(this);
                            var rect = range.getBoundingClientRect().toJSON();
                            range.detach();
                            return rect;
                        } else {
                            return this.getBoundingClientRect().toJSON();
                        }
                    }
                """,
                "returnByValue": True,
            },
        )
        return response
    except:
        return {"result": {"subtype": "error"}}


def fetch_page_accessibility_tree(
    info: BrowserInfo,
    browser,
    current_viewport_only: bool,
) -> AccessibilityTree:
    accessibility_tree: AccessibilityTree = browser.execute_cdp_cmd(
        "Accessibility.getFullAXTree", {}
    )["nodes"]

    seen_ids = set()
    _accessibility_tree = []
    for node in accessibility_tree:
        if node["nodeId"] not in seen_ids:
            _accessibility_tree.append(node)
            seen_ids.add(node["nodeId"])
    accessibility_tree = _accessibility_tree

    nodeid_to_cursor = {}
    for cursor, node in enumerate(accessibility_tree):
        nodeid_to_cursor[node["nodeId"]] = cursor
        if "backendDOMNodeId" not in node:
            node["union_bound"] = None
            continue
        backend_node_id = str(node["backendDOMNodeId"])
        if node["role"]["value"] == "RootWebArea":
            node["union_bound"] = [0.0, 0.0, 10.0, 10.0]
        else:
            response = get_bounding_client_rect(
                browser, backend_node_id
            )
            if response.get("result", {}).get("subtype", "") == "error":
                node["union_bound"] = None
            else:
                x = response["result"]["value"]["x"]
                y = response["result"]["value"]["y"]
                width = response["result"]["value"]["width"]
                height = response["result"]["value"]["height"]
                node["union_bound"] = [x, y, width, height]

    if current_viewport_only:

        def remove_node_in_graph(node: AccessibilityTreeNode) -> None:
            nodeid = node["nodeId"]
            node_cursor = nodeid_to_cursor[nodeid]
            parent_nodeid = node["parentId"]
            children_nodeids = node["childIds"]
            parent_cursor = nodeid_to_cursor[parent_nodeid]
            assert (
                accessibility_tree[parent_cursor].get("parentId", "Root")
                is not None
            )
            index = accessibility_tree[parent_cursor]["childIds"].index(
                nodeid
            )
            accessibility_tree[parent_cursor]["childIds"].pop(index)
            for child_nodeid in children_nodeids:
                accessibility_tree[parent_cursor]["childIds"].insert(
                    index, child_nodeid
                )
                index += 1
            for child_nodeid in children_nodeids:
                child_cursor = nodeid_to_cursor[child_nodeid]
                accessibility_tree[child_cursor][
                    "parentId"
                ] = parent_nodeid
            accessibility_tree[node_cursor]["parentId"] = "[REMOVED]"

        config = info["config"]
        for node in accessibility_tree:
            if not node["union_bound"]:
                remove_node_in_graph(node)
                continue

            [x, y, width, height] = node["union_bound"]

            if width == 0 or height == 0:
                remove_node_in_graph(node)
                continue

            in_viewport_ratio = get_element_in_viewport_ratio(
                elem_left_bound=float(x),
                elem_top_bound=float(y),
                width=float(width),
                height=float(height),
                config=config,
            )

            if in_viewport_ratio < IN_VIEWPORT_RATIO_THRESHOLD:
                remove_node_in_graph(node)

        accessibility_tree = [
            node
            for node in accessibility_tree
            if node.get("parentId", "Root") != "[REMOVED]"
        ]

    return accessibility_tree


def parse_accessibility_tree(
    accessibility_tree: AccessibilityTree,
) -> tuple[str, dict[str, Any]]:
    node_id_to_idx = {}
    for idx, node in enumerate(accessibility_tree):
        node_id_to_idx[node["nodeId"]] = idx

    nodeIdToTreeIds = {}
    def dfs(idx: int, depth: int, parent_name: str) -> str:
        tree_str = ""
        node = accessibility_tree[idx]
        indent = "\t" * depth
        valid_node = True
        try:
            role = node["role"]["value"]
            name = node["name"]["value"]
            node_str = f"{role} {repr(name)}"
            if not name.strip() or role in ['gridcell'] or (name.strip() in parent_name and role in ['StaticText', 'heading', 'image', 'generic']):
                valid_node = False
            else:
                properties = []
                for property in node.get("properties", []):
                    try:
                        if property["name"] in IGNORED_ACTREE_PROPERTIES:
                            continue
                        properties.append(
                            f'{property["name"]}: {property["value"]["value"]}'
                        )
                    except KeyError:
                        pass

                if properties:
                    node_str += " " + " ".join(properties)

            if valid_node:
                nodeIdToTreeIds[len(nodeIdToTreeIds)+1] = node
                tree_str += f"{indent}[{len(nodeIdToTreeIds)}] {node_str}"

        except:
            valid_node = False

        for _, child_node_id in enumerate(node["childIds"]):
            if child_node_id not in node_id_to_idx:
                continue
            if len(nodeIdToTreeIds) > 300:
                break
            child_depth = depth + 1 if valid_node else depth
            curr_name = name if valid_node else parent_name
            child_str = dfs(
                node_id_to_idx[child_node_id], child_depth, curr_name
            )
            if child_str.strip():
                if tree_str.strip():
                    tree_str += "\n"
                tree_str += child_str

        return tree_str

    tree_str = dfs(0, 0, 'root')
    return tree_str, nodeIdToTreeIds


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_information(action_str):
    """Parse an action string into (action_key, info)."""
    action_str = action_str.strip()

    # Strip an optional Action: prefix.
    if action_str.lower().startswith("action:"):
        action_str = action_str[len("action:"):].strip()


    first_space_idx = action_str.find(' ')
    if first_space_idx == -1:
        action_key = action_str.lower()
        args_str_raw = ""
    else:
        action_key = action_str[:first_space_idx].lower()
        args_str_raw = action_str[first_space_idx + 1:].strip()

    info = {}

    if action_key in ['click', 'dbclick']:
        match = re.match(r'^(?:\[(\d+)\]|(\d+))', args_str_raw)
        if match:
            number_str = match.group(1) or match.group(2)
            info['number'] = int(number_str)
        else:
            raise ValueError(f"Invalid {action_key} action format: '{action_str}'. Expected '{action_key} [number]' (e.g., '{action_key} [15]') or '{action_key} number' (e.g., '{action_key} 15').")

    elif action_key == 'type':
        # Accept: [num] "text" or [num] text
        match = re.match(r'^\[(\d+)\]\s*(?:"(.*?)"|(.+))', args_str_raw, re.DOTALL)
        if match:
            info['number'] = int(match.group(1))
            if match.group(2) is not None:
                info['content'] = match.group(2)
            else:
                info['content'] = match.group(3).strip()
        else:
            try:
                parts = args_str_raw.split(' ', 1)
                if len(parts) == 2:
                    num_part = parts[0]
                    content_part_raw = parts[1]
                    info['number'] = int(num_part)
                    if content_part_raw.startswith('"') and content_part_raw.endswith('"'):
                        info['content'] = content_part_raw[1:-1]
                    elif content_part_raw.startswith("'") and content_part_raw.endswith("'"):
                        info['content'] = content_part_raw[1:-1]
                    else:
                        info['content'] = content_part_raw.strip()
                    logging.warning(f"Using fallback for type action: '{action_str}'. Expected 'type [number] [content]' but got simple format.")
                else:
                    raise ValueError("Not enough parts for type action.")
            except (ValueError, IndexError):
                raise ValueError(f"Invalid type action format: '{action_str}'. Expected 'type [number] \"content\"' (e.g., 'type [5] \"some text\"') or 'type number \"content\"'.")


    elif action_key == 'scroll':
        match = re.match(r'^\[(WINDOW|\d+)\]\s*\[(up|down)\]', args_str_raw, re.IGNORECASE)
        if match:
            number_or_window = match.group(1)
            info['number'] = int(number_or_window) if number_or_window.isdigit() else number_or_window.upper()
            info['content'] = match.group(2).lower()
        else:
            try:
                parts = args_str_raw.split(' ', 1)
                num_or_win_part = parts[0]
                direction_part = parts[1] if len(parts) > 1 else ""

                info['number'] = int(num_or_win_part) if num_or_win_part.isdigit() else num_or_win_part.upper()
                info['content'] = direction_part.lower().strip()
                if info['content'] not in ['up', 'down']:
                     raise ValueError("Invalid scroll direction in fallback.")
                logging.warning(f"Using fallback for scroll action: '{action_str}'. Expected 'scroll [number/WINDOW] [up|down]' but got simple format.")
            except (ValueError, IndexError):
                raise ValueError(f"Invalid scroll action format: '{action_str}'. Expected 'scroll [number/WINDOW] [up|down]' (e.g., 'scroll [WINDOW] [down]') or 'scroll number/WINDOW direction'.")

    elif action_key in ['answer', 'stop']:
        match = re.match(r'^\[(.*?)\]', args_str_raw, re.DOTALL)
        if match:
            info['content'] = match.group(1).strip()
        else:
            info['content'] = args_str_raw.strip()
            logging.warning(f"Using fallback for {action_key} action: '{action_str}'. Expected '{action_key} [content]' but got simple content.")

    elif action_key in ['wait', 'go_back', 'restart']: # Changed 'goback' to 'go_back' for consistency with prompt
        if args_str_raw.strip():
            logging.warning(f"Action '{action_key}' received unexpected arguments: '{args_str_raw}'. Ignoring.")
        info = {}
    else:
        raise ValueError(f"Unknown action key or invalid format: '{action_key}'. Full action string: '{action_str}'")

    return action_key, info
def print_message(json_object, save_dir=None, is_v=False):
    remove_b64code_obj = []
    for obj in json_object:
        if obj['role'] != 'user':
            logging.info(obj)
            remove_b64code_obj.append(obj)
        else:
            if type(obj['content']) == str:
                logging.info(obj)
                remove_b64code_obj.append(obj)
            else:
                print_obj = {
                    'role': obj['role'],
                    'content': obj['content']
                }
                for item in print_obj['content']:
                    if item['type'] == 'image_url':
                        item['image_url'] =  {"url": "data:image/png;base64,{b64_img}"}
                logging.info(print_obj)
                remove_b64code_obj.append(print_obj)
    if save_dir:
        with open(os.path.join(save_dir, 'interact_messages.json'), 'w', encoding='utf-8') as fw:
            json.dump(remove_b64code_obj, fw, indent=2)


def get_webarena_accessibility_tree(browser, save_file=None):
    browser_info = fetch_browser_info(browser)
    accessibility_tree = fetch_page_accessibility_tree(browser_info, browser, current_viewport_only=True)
    print ('Accessibility tree nodes:', len(accessibility_tree))
    content, obs_nodes_info = parse_accessibility_tree(accessibility_tree)
    if save_file:
        with open(save_file + '.json', 'w', encoding='utf-8') as fw:
            json.dump(obs_nodes_info, fw, indent=2)
        with open(save_file + '.txt', 'w', encoding='utf-8') as fw:
            fw.write(content)


    return content, obs_nodes_info


def get_pdf_retrieval_ans_from_assistant(client, pdf_path, task):
    logging.info("You download a PDF file that will be retrieved using the Assistant API.")
    file = client.files.create(
        file=open(pdf_path, "rb"),
        purpose='assistants'
    )
    logging.info("Create assistant...")
    assistant = client.beta.assistants.create(
        instructions="You are a helpful assistant that can analyze the content of a PDF file and give an answer that matches the given task, or retrieve relevant content that matches the task.",
        model="gpt-4-1106-preview",
        tools=[{"type": "retrieval"}],
        file_ids=[file.id]
    )
    thread = client.beta.threads.create()
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=task,
        file_ids=[file.id]
    )
    print(thread)
    print(message)
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id
    )
    while True:
        run_status = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        if run_status.status == 'completed':
            break
        time.sleep(2)
    messages = client.beta.threads.messages.list(thread_id=thread.id)
    messages_text = messages.data[0].content[0].text.value
    file_deletion_status = client.beta.assistants.files.delete(
        assistant_id=assistant.id,
        file_id=file.id
    )
    logging.info(file_deletion_status)
    assistant_deletion_status = client.beta.assistants.delete(assistant.id)
    logging.info(assistant_deletion_status)
    return messages_text
