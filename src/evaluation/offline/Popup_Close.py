import os
import sys
import json
import base64
import re
import asyncio
import aiofiles
from tqdm.asyncio import tqdm_asyncio
from openai import AsyncOpenAI

Model_name = "UI-TARs"  # 模型名称

# ===== 配置项 =====
# 请确保这里的路径和模型名是正确的
TEST_JSON_PATH = "/code/CogReasoner/Test/Popup_close.json"
MODEL_NAME = "qwen2vl"
MAX_SAMPLE = 60  # 您可以根据需要调整测试样本数，None表示测试全部
MAX_CONCURRENT_REQUESTS = 5
ACCURACY_PRINT_INTERVAL = 10
OUTPUT_JSON_PATH = f"/code/CogReasoner/Code/Evalaute/Result/Test-{Model_name}-Popup_close.json"

# ===== 初始化 OpenAI 客户端 =====
client = AsyncOpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8080/v1",
)

# ===== 关键修改 1: 新的答案提取函数 =====
def extract_action_from_output(text: str):
    """
    从模型的详细输出中提取最终的行动指令。
    它会寻找 "Action: action_name [node_id]" 这样的格式。
    
    Args:
        text (str): 模型的原始输出字符串。
        
    Returns:
        str or None: 格式化后的答案，如 "CLICK(41)"，如果找不到则返回 None。
    """
    # 正则表达式寻找 "Action: "，后面跟着一个单词(action)和一个用方括号括起来的数字(node_id)
    # re.IGNORECASE 使得 "click" 和 "Click" 都能匹配
    match = re.search(r"Action:\s+(\w+)\s+\[(\d+)\]", text, re.IGNORECASE)
    
    if match:
        action = match.group(1).upper()  # 提取动作并转为大写, e.g., "CLICK"
        node_id = match.group(2)         # 提取节点ID, e.g., "41"
        return f"{action}({node_id})"    # 组合成与Ground Truth一致的格式
        
    return None # 如果没有找到匹配项，返回 None

# ===== 异步处理单个样本 =====
async def process_item(index, item, sem, stats):
    async with sem:
        image_paths = item["images"]
        prompt = item["messages"][0]["content"]
        prompt = prompt[prompt.find("OBSERVATION:"):]
        gt_json_str = item["messages"][-1]["content"]
        try:
            gt_data = json.loads(gt_json_str) 
            action = gt_data.get('ACTION', gt_data.get('action', '')).upper()
            node_id = gt_data.get('NODE_ID', gt_data.get('node_id'))
            gt_answer = f"{action}({node_id})"
        except (json.JSONDecodeError, TypeError, AttributeError):
            print(f"警告: 无法解析 Ground Truth: {gt_json_str}")
            gt_answer = "INVALID_GT_FORMAT"

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
                # 如果图片找不到，记录错误并跳过这个样本
                error_msg = f"[ERROR] Image not found at {path}"
                print(error_msg)
                return {
                    "images": image_paths,
                    "ground_truth": gt_answer,
                    "prediction": None,
                    "match": False,
                    "raw_model_output": error_msg
                }

        # 构造消息 (这里的prompt构造逻辑保持不变)
        messages = [
            {"role": "system", "content": "You are a helpful assistant that analyzes UI screenshots and determines the next action."},
            {
                "role": "user",
                "content": image_contents + [
                    {
                        "type": "text",
                        "text": "Determine the single, most direct action required to close or dismiss the popup.",
                        # + prompt.strip() + "\n\n## Output Format:\nPlease output only the action in the format: `Action: [action_name] [node_id]`
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

        # <--- 关键修改 3: 使用新的提取函数 ---
        pred_answer = extract_action_from_output(pred_text)
        
        # 比较预测答案和真实答案
        match = pred_answer == gt_answer

        # 更新统计数据
        stats["total"] += 1
        stats["correct"] += int(match)

        if stats["total"] > 0 and stats["total"] % ACCURACY_PRINT_INTERVAL == 0:
            acc = stats["correct"] / stats["total"] * 100
            print(f"\n📊 Step {stats['total']}: Accuracy = {acc:.2f}%\n")

        # 返回结果字典
        return {
            "images": image_paths,
            "ground_truth": gt_answer,
            "prediction": pred_answer,
            "match": match,
            "raw_model_output": pred_text
        }

# ===== 主函数 (少量修改以适应新逻辑) =====
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
    
    # 过滤掉因为图片找不到等原因未能处理的样本
    valid_results = [r for r in results if r is not None]
    if not valid_results:
        print("\n❌ No valid samples were processed. Evaluation cannot be completed.")
        return

    # 重新计算最终准确率，以防有样本被跳过
    final_total = stats["total"]
    final_correct = stats["correct"]
    accuracy = (final_correct / final_total * 100) if final_total > 0 else 0
    
    # 找出错误的例子
    errors = [r for r in valid_results if not r["match"]]

    # 准备输出文件
    output = {
        "model_name": Model_name,
        "metrics": {
            "total_processed": final_total,
            "correct": final_correct,
            "accuracy": accuracy
        },
        "results": valid_results, # 保存所有有效结果
        "errors": errors
    }

    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # 控制台输出摘要
    print(f"\n✅ Evaluation Complete")
    print(f"🎯 Accuracy: {accuracy:.2f}% ({final_correct}/{final_total})")
    print(f"📁 Results saved to: {OUTPUT_JSON_PATH}")

    # <--- 关键修改 4: 修正错误样本的打印逻辑 ---
    if errors:
        print("\n❌ Sample Errors (up to 5):")
        for r in errors[:5]:
            print(f"- Images       : {', '.join(r['images'])}")
            print(f"  Ground Truth : {r['ground_truth']}")
            print(f"  Prediction   : {r['prediction']}")
            # 为了简洁，可以只打印raw_model_output的最后一部分
            raw_output_snippet = r['raw_model_output'][-200:].replace('\n', ' ')
            print(f"  Raw Output...: ...{raw_output_snippet}\n")

    await client.aclose()

# ===== 启动入口 =====
if __name__ == "__main__":
    asyncio.run(main())
    # 移除 sys.exit(0) 以允许正常的异步清理