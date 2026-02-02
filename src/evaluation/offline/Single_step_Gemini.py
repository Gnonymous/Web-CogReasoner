import os
import sys
import json
import base64
import re
import asyncio
import aiofiles
from tqdm.asyncio import tqdm_asyncio
from openai import AsyncOpenAI

# ===================================================================
# ===== 1. 合并后的配置项 =====
# ===================================================================

# --- 模型和输出配置 (来自你的Gemini脚本) ---
Model_name = "Gemini"  # 用于输出文件的模型名称
# 请确认这是你想要用于此任务的Gemini模型ID
MODEL_NAME = "gemini-2.5-pro"  # 推荐使用多模态的 gemini-pro-vision

# --- 任务配置 (来自你的脚本A) ---
TEST_JSON_PATH = "/code/CogReasoner/Test/MultiStep_Selected_OnePerSite_step01_FinalAction.json"
MAX_SAMPLE = 70  # 您可以根据需要调整测试样本数，None表示测试全部
MAX_CONCURRENT_REQUESTS = 5 # 如果遇到 Gemini 的速率限制错误，可以调低此值
ACCURACY_PRINT_INTERVAL = 10
OUTPUT_JSON_PATH = f"/code/CogReasoner/Code/Evalaute/Result/Test-{Model_name}-Single_Step.json"

# ===================================================================
# ===== 2. 初始化 Gemini 客户端 (关键修改) =====
# ===================================================================
# 我们使用你提供的Gemini配置来初始化客户端
# 这会通过一个兼容层将OpenAI格式的请求发往Google Gemini
client = AsyncOpenAI(
    # <--- 修改: 请确保你的API密钥是有效的
    api_key=os.getenv("GEMINI_API_KEY", ""), # 例如: "YOUR_GEMINI_API_KEY"
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/" # 基础URL通常是这个
)
# 注意: Gemini的URL通常不包含 /openai/ 后缀，除非你使用了特殊的代理。
# 标准的Gemini API是 /v1beta/models/gemini-pro-vision:generateContent
# 但如果你使用的库或代理需要 /openai/，请保留它。为保险起见，我使用了你提供的URL格式但去掉了末尾的openai/
# 更新：为适配 openai-python 库，通常会带一个 /models/ 后缀
# client.base_url = "https://generativelanguage.googleapis.com/v1beta/models/"  <-- 这可能是更兼容的格式
# 但我们先按你提供的配置来，如果报错再调整。

# ===================================================================
# ===== 3. 核心函数 (来自脚本 A, 完全保留) =====
# ===================================================================

def extract_action(text: str):
    """从模型输出中提取标准化的动作字符串。
    这个版本可以处理 CLICK 和带输入值的 TYPE。"""
    if not text:
        return None
    # 优先匹配带输入值的TYPE动作
    type_match = re.search(r"Action:\s+type\s+\[(\d+)\]\s+\((.*?)\)", text, re.IGNORECASE)
    if type_match:
        node_id = type_match.group(1)
        value = type_match.group(2)
        return f"TYPE({node_id}, {value})"
    # 匹配简单的动作，如CLICK
    simple_match = re.search(r"Action:\s+(\w+)\s+\[(\d+)\]", text, re.IGNORECASE)
    if simple_match:
        action = simple_match.group(1).upper()
        node_id = simple_match.group(2)
        return f"{action}({node_id})"
    return None

def parse_ground_truth(gt_content: str):
    """解析可能包含多个答案的 Ground Truth 字符串。"""
    action_parts = gt_content.split(';')
    parsed_actions = [extract_action(part.strip()) for part in action_parts]
    return [action for action in parsed_actions if action is not None]

def compare_actions(prediction: str, ground_truth_list: list) -> bool:
    """智能比较函数，特别处理 TYPE 动作。"""
    if not prediction:
        return False

    pred_match = re.match(r"(\w+)\((\d+)", prediction)
    if not pred_match:
        return False
    pred_action_type = pred_match.group(1)
    pred_node_id = pred_match.group(2)

    for gt_action in ground_truth_list:
        # 规则 2: 完全匹配
        if prediction == gt_action:
            return True
        # 规则 3: 特殊规则 for TYPE
        if pred_action_type == "TYPE" and gt_action.startswith("TYPE("):
            gt_match = re.match(r"TYPE\((\d+)", gt_action)
            if gt_match and pred_node_id == gt_match.group(1):
                return True
    return False

# ===================================================================
# ===== 4. 异步处理函数 (保留脚本A的逻辑) =====
# ===================================================================

async def process_item(index, item, sem, stats):
    async with sem:
        image_paths = item["images"]
        prompt = item["messages"][0]["content"]
        gt_json_str = item["messages"][-1]["content"]

        # 使用脚本A的多答案解析逻辑
        gt_answers_list = parse_ground_truth(gt_json_str)

        if not gt_answers_list:
            print(f"⚠️ 无法解析 Ground Truth: {gt_json_str}")
            # 返回一个与主结构一致的字典，避免后续处理出错
            return {
                "images": image_paths, "prompt": prompt, "ground_truth": "INVALID_GT_FORMAT",
                "prediction": None, "match": False, "raw_model_output": "Ground truth format is invalid."
            }

        image_contents = []
        for path in image_paths:
            try:
                async with aiofiles.open(path, "rb") as f:
                    content = await f.read()
                encoded_image = base64.b64encode(content).decode("utf-8")
                # 使用标准的OpenAI格式，兼容层会处理它
                image_contents.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded_image}"}
                })
            except FileNotFoundError:
                error_msg = f"[ERROR] Image not found at {path}"
                print(error_msg)
                return {
                    "images": image_paths, "prompt": prompt, "ground_truth": ";".join(gt_answers_list),
                    "prediction": None, "match": False, "raw_model_output": error_msg
                }

        # 使用脚本A的任务提示 (Prompt)
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

        try:
            # 这里的API调用代码无需改变，因为我们已经配置了client指向Gemini
            response = await client.chat.completions.create(
                model=MODEL_NAME, # 使用Gemini的模型名称
                messages=messages,
                temperature=0.1,
                top_p=0.95,
                max_tokens=2048,
            )
            # Gemini的响应可能需要不同的解析，但兼容层通常会处理好
            pred_text = response.choices[0].message.content.strip()
        except Exception as e:
            pred_text = f"[ERROR] {str(e)}"

        # 使用脚本A的答案提取和智能比较逻辑
        pred_answer = extract_action(pred_text)
        match = compare_actions(pred_answer, gt_answers_list)
        
        # 统计逻辑保持不变
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
# ===== 5. 主函数 (保留脚本A的逻辑) =====
# ===================================================================

async def main():
    with open(TEST_JSON_PATH, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    if MAX_SAMPLE is not None and MAX_SAMPLE > 0:
        test_data = test_data[:MAX_SAMPLE]

    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    stats = {"total": 0, "correct": 0}
    # 任务创建时不需要传递client，因为它在全局作用域中
    tasks = [process_item(i, item, sem, stats) for i, item in enumerate(test_data)]

    print(f"\n🚀 开始评估 {len(tasks)} 个样本，使用模型 '{MODEL_NAME}'...\n")
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

    await client.aclose()

# ===================================================================
# ===== 6. 启动入口 (无变化) =====
# ===================================================================

if __name__ == "__main__":
    if "YOUR_GEMINI_API_KEY" in client.api_key:
        print("错误: 请在脚本中填写您的 Gemini API Key。")
        sys.exit(1)
    asyncio.run(main())