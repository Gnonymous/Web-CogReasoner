import os
import json
import base64
import re
import asyncio
import aiofiles
import sys  # <--- 新增: 导入sys模块以支持sys.exit()
from tqdm.asyncio import tqdm_asyncio
from openai import AsyncOpenAI

Model_name = "Stage_2"  # 模型名称

# ===== 配置项 =====
TEST_JSON_PATH = "/code/CogReasoner/Test/Action_Prediction.json"  # 测试集 JSON 路径
MODEL_NAME = "qwen2vl"  # 使用的模型名称
MAX_SAMPLE = 44  # 测试样本数
MAX_CONCURRENT_REQUESTS = 5  # 最大并发数
ACCURACY_PRINT_INTERVAL = 10  # 每多少步打印一次准确率``
OUTPUT_JSON_PATH = f"/code/CogReasoner/Code/Evalaute/Result/Test-{Model_name}-Action_Prediction.json"  # 推理结果保存路径

# ===== 初始化 OpenAI 客户端（对接 vLLM API） =====
client = AsyncOpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8080/v1",
)

# ===== 提取模型输出的选项，如 G、A、B等 =====
# 此函数无需修改，它能正确地从模型输出中提取第一个字母选项。
def extract_answer_letter(text):
    match = re.search(r"\b([A-Z])\b", text.strip(), re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None

# ===== 异步处理单个样本 =====
# ***** 主要修改区域 *****
async def process_item(index, item, sem, stats):
    async with sem:
        image_paths = item["images"]
        # --- 修改开始 ---
        # 1. 获取可能包含多个选项的正确答案字符串, e.g., "H,G,I,E"
        gt_answer_str = item["messages"][-1]["content"].strip().upper()
        # 2. 将答案字符串按逗号分割，创建一个包含所有正确选项的集合
        possible_gt_answers = {opt.strip() for opt in gt_answer_str.split(',')}
        # --- 修改结束 ---
        prompt = item["messages"][0]["content"]

        # 编码所有图像为 base64 并构造 image_contents
        image_contents = []
        for path in image_paths:
            async with aiofiles.open(path, "rb") as f:
                content = await f.read()
            encoded_image = base64.b64encode(content).decode("utf-8")
            image_contents.append({
                "type": "image_url",
                "image_url": {"url": f"data:image;base64,{encoded_image}"}
            })

        # 构造消息
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
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
                max_tokens=512,
            )
            pred_text = response.choices[0].message.content.strip()
        except Exception as e:
            pred_text = f"[ERROR] {str(e)}"

        pred_answer = extract_answer_letter(pred_text)
        
        # --- 修改开始 ---
        # 3. 检查预测的单个答案是否存在于正确答案集合中
        match = pred_answer is not None and pred_answer in possible_gt_answers
        # --- 修改结束 ---

        stats["total"] += 1
        stats["correct"] += int(match)

        if stats["total"] % ACCURACY_PRINT_INTERVAL == 0:
            acc = stats["correct"] / stats["total"] * 100
            print(f"\n📊 Step {stats['total']}: Accuracy = {acc:.2f}%\n")

        return {
            "images": image_paths,
            "ground_truth": gt_answer_str, # 返回原始的正确答案字符串
            "prediction": pred_answer,
            "match": match,
            "raw_model_output": pred_text
        }

# ===== 主函数 =====
async def main():
    with open(TEST_JSON_PATH, "r", encoding="utf-8") as f:
        test_data = json.load(f)[:MAX_SAMPLE]

    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    stats = {"total": 0, "correct": 0}
    tasks = [process_item(i, item, sem, stats) for i, item in enumerate(test_data)]

    print(f"\n🚀 Starting evaluation of {len(tasks)} samples...\n")
    results = await tqdm_asyncio.gather(*tasks)

    accuracy = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
    errors = [r for r in results if not r["match"]]

    # 写入输出
    output = {
        "metrics": {
            "total": stats["total"],
            "correct": stats["correct"],
            "accuracy": accuracy
        },
        "errors": errors
    }

    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # 控制台输出摘要
    print(f"\n✅ Evaluation Complete")
    print(f"🎯 Accuracy: {accuracy:.2f}%")
    print(f"📁 Results saved to: {OUTPUT_JSON_PATH}")

    print("\n❌ Sample Errors (up to 5):")
    for r in errors[:5]:
        # --- 修复: 'image' 键应为 'images' ---
        print(f"- Images       : {r['images']}")
        print(f"  Ground Truth : {r['ground_truth']}")
        print(f"  Prediction   : {r['prediction']}")
        print(f"  Raw Output   : {r['raw_model_output']}\n")

    await client.aclose()  # ✅ 释放连接池

# ===== 启动入口 =====
if __name__ == "__main__":
    asyncio.run(main())
    sys.exit(0)  # ✅ 强制退出，防止异步底层未回收导致挂起