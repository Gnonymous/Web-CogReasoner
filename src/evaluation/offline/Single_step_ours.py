import os
import sys
import json
import base64
import re
import asyncio
import aiofiles
from tqdm.asyncio import tqdm_asyncio
from openai import AsyncOpenAI

Model_name = "CogReasoner"  # 模型名称

# ===== 配置项 =====
# 请确保这里的路径和模型名是正确的
TEST_JSON_PATH = "/code/CogReasoner/Test/MultiStep_Selected_OnePerSite_step01_FinalAction.json"
MODEL_NAME = "qwen2vl"
MAX_SAMPLE = 70  # 您可以根据需要调整测试样本数，None表示测试全部
MAX_CONCURRENT_REQUESTS = 5
ACCURACY_PRINT_INTERVAL = 10
OUTPUT_JSON_PATH = f"/code/CogReasoner/Code/Evalaute/Result/Test-{Model_name}-Single_Step.json"

# ===== 初始化 OpenAI 客户端 =====
client = AsyncOpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8080/v1",
)

# ===== 答案提取函数 (无变化) =====
def extract_action(text: str):
    if not text:
        return None
    type_match = re.search(r"Action:\s+type\s+\[(\d+)\]\s+\[(.*?)\]", text, re.IGNORECASE)
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

# ===== 解析多答案的 Ground Truth (无变化) =====
def parse_ground_truth(gt_content: str):
    action_parts = gt_content.split(';')
    parsed_actions = [extract_action(part.strip()) for part in action_parts]
    return [action for action in parsed_actions if action is not None]

# ===== 关键修改 1: 新增智能比较函数 =====
def compare_actions(prediction: str, ground_truth_list: list) -> bool:
    """
    智能比较预测动作和真实动作列表。

    规则:
    1. 如果预测为空，则不匹配。
    2. 对于任何动作，如果预测与列表中的任何一个真实动作完全相同，则匹配。
    3. **特殊规则**: 如果预测动作和真实动作都是 TYPE 类型，
       只要它们的 node_id 相同，就认为它们匹配，忽略输入的具体文本。
    
    Args:
        prediction (str): 标准化后的预测动作，例如 "TYPE(6, LYHNCNCT)"。
        ground_truth_list (list): 标准化后的真实动作列表，例如 ["TYPE(6, LYNCT)"]。

    Returns:
        bool: 如果匹配则返回 True，否则返回 False。
    """
    if not prediction:
        return False

    # 预先解析预测动作的类型和ID
    pred_match = re.match(r"(\w+)\((\d+)", prediction)
    if not pred_match:
        return False # 预测格式不正确
    pred_action_type = pred_match.group(1)
    pred_node_id = pred_match.group(2)

    for gt_action in ground_truth_list:
        # 规则 2: 完全匹配总是正确的
        if prediction == gt_action:
            return True

        # 规则 3: 对 TYPE 动作的特殊处理
        if pred_action_type == "TYPE" and gt_action.startswith("TYPE("):
            gt_match = re.match(r"TYPE\((\d+)", gt_action)
            if gt_match:
                gt_node_id = gt_match.group(1)
                if pred_node_id == gt_node_id:
                    return True # Node ID 匹配，判定为正确
    
    return False

# ===== 异步处理单个样本 (少量修改) =====
async def process_item(index, item, sem, stats):
    async with sem:
        image_paths = item["images"]
        prompt = item["messages"][0]["content"]
        
        gt_json_str = item["messages"][-1]["content"]
        gt_answers_list = parse_ground_truth(gt_json_str)

        if not gt_answers_list:
            print(f"⚠️ 无法解析 Ground Truth: {gt_json_str}")
            return {
                "images": image_paths,
                "prompt": prompt,
                "ground_truth": "INVALID_GT_FORMAT",
                "prediction": None,
                "match": False,
                "raw_model_output": "Ground truth format is invalid."
            }

        image_contents = []
        for path in image_paths:
            try:
                async with aiofiles.open(path, "rb") as f:
                    content = await f.read()
                encoded_image = base64.b64encode(content).decode("utf-8")
                image_contents.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image;base64,{encoded_image}"}
                })
            except FileNotFoundError:
                error_msg = f"[ERROR] Image not found at {path}"
                print(error_msg)
                return {
                    "images": image_paths,
                    "prompt": prompt,
                    "ground_truth": ";".join(gt_answers_list),
                    "prediction": None,
                    "match": False,
                    "raw_model_output": error_msg
                }

        messages = [
            {
                "role": "user",
                "content": image_contents + [
                    {
                        "type": "text",
                        "text": prompt.strip(),
                    }
                ],
            },
        ]

        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.1,
                top_p=0.95,
                max_tokens=2048,
            )
            pred_text = response.choices[0].message.content.strip()
        except Exception as e:
            pred_text = f"[ERROR] {str(e)}"

        pred_answer = extract_action(pred_text)
        
        # <--- 关键修改 2: 使用新的智能比较函数 ---
        match = compare_actions(pred_answer, gt_answers_list)
        
        # 更新统计数据
        stats["total"] += 1
        stats["correct"] += int(match)

        if stats["total"] > 0 and stats["total"] % ACCURACY_PRINT_INTERVAL == 0:
            acc = stats["correct"] / stats["total"] * 100
            print(f"\n📊 Step {stats['total']}: Accuracy = {acc:.2f}%\n")

        # 返回结果字典
        return {
            "images": image_paths,
            "prompt": prompt,
            "ground_truth": ";".join(gt_answers_list),
            "prediction": pred_answer,
            "match": match,
            "raw_model_output": pred_text
        }


# ===== 主函数 (无变化) =====
async def main():
    with open(TEST_JSON_PATH, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    
    if MAX_SAMPLE is not None and MAX_SAMPLE > 0:
        test_data = test_data[:MAX_SAMPLE]

    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    stats = {"total": 0, "correct": 0}
    tasks = [process_item(i, item, sem, stats) for i, item in enumerate(test_data)]

    print(f"\n🚀 Starting evaluation of {len(tasks)} samples with model '{MODEL_NAME}'...\n")
    results = await tqdm_asyncio.gather(*tasks)
    
    valid_results = [r for r in results if r is not None]
    if not valid_results:
        print("\n❌ No valid samples were processed. Evaluation cannot be completed.")
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

    print(f"\n✅ Evaluation Complete")
    print(f"🎯 Accuracy: {accuracy:.2f}% ({final_correct}/{final_total})")
    print(f"📁 Results saved to: {OUTPUT_JSON_PATH}")

    if errors:
        print("\n❌ Sample Errors (up to 5):")
        for r in errors[:5]:
            print(f"- Images       : {', '.join(r['images'])}")
            print(f"  Ground Truth : {r['ground_truth']}")
            print(f"  Prediction   : {r['prediction']}")
            raw_output_snippet = r['raw_model_output'].replace('\n', ' ')
            if len(raw_output_snippet) > 200:
                raw_output_snippet = "..." + raw_output_snippet[-200:]
            print(f"  Raw Output   : {raw_output_snippet}\n")

    await client.aclose()

# ===== 启动入口 (无变化) =====
if __name__ == "__main__":
    asyncio.run(main())