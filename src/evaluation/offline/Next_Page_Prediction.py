import os
import json
import base64
import re
import asyncio
import aiofiles
from tqdm.asyncio import tqdm_asyncio
from openai import AsyncOpenAI

Model_name = "OpenWebVoyager"  # 模型名称

# ===== 配置项 =====
TEST_JSON_PATH = "/code/CogReasoner/Test/Next_Page_Prediction_100.json"  # 测试集 JSON 路径
MODEL_NAME = "qwen2vl"  # 使用的模型名称
MAX_SAMPLE = 93  # 测试样本数
MAX_CONCURRENT_REQUESTS = 10  # 最大并发数
ACCURACY_PRINT_INTERVAL = 10  # 每多少步打印一次准确率
OUTPUT_JSON_PATH = f"/code/CogReasoner/Code/Evalaute/Result/Test-{Model_name}-Next_Page_Prediction_100.json"  # 推理结果保存路径

# ===== 初始化 OpenAI 客户端（对接 vLLM API） =====
client = AsyncOpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8080/v1",
)

# ===== 提取模型输出的选项，如 G、A、B等 =====
def extract_answer_letter(text):
    match = re.search(r"\b([A-H])\b", text.strip(), re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None

# ===== 异步处理单个样本 =====
async def process_item(index, item, sem, stats):
    async with sem:
        image_path = item["images"][0]
        gt_answer = item["messages"][-1]["content"].strip().upper()
        prompt = item["messages"][0]["content"]

        # 编码图像
        async with aiofiles.open(image_path, "rb") as f:
            content = await f.read()
        encoded_image = base64.b64encode(content).decode("utf-8")
        image_data_uri = f"data:image;base64,{encoded_image}"

        try:
            # 推理请求
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_data_uri}},
                            {
                                "type": "text",
                                "text": prompt + "Directly give the answer letter (A, B, C, D, E, F, G, H) without any explanation.",
                            },
                        ],
                    },
                ],
                temperature=0.1,
                top_p=0.95,
                max_tokens=512,
            )
            pred_text = response.choices[0].message.content.strip()
        except Exception as e:
            pred_text = f"[ERROR] {str(e)}"

        pred_answer = extract_answer_letter(pred_text)
        match = pred_answer == gt_answer

        stats["total"] += 1
        stats["correct"] += int(match)

        if stats["total"] % ACCURACY_PRINT_INTERVAL == 0:
            acc = stats["correct"] / stats["total"] * 100
            print(f"\n📊 Step {stats['total']}: Accuracy = {acc:.2f}%\n")

        return {
            "image": image_path,
            "ground_truth": gt_answer,
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

    accuracy = stats["correct"] / stats["total"] * 100
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

    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # 控制台输出摘要
    print(f"\n✅ Evaluation Complete")
    print(f"🎯 Accuracy: {accuracy:.2f}%")
    print(f"📁 Results saved to: {OUTPUT_JSON_PATH}")

    print("\n❌ Sample Errors (up to 5):")
    for r in errors[:5]:
        print(f"- Image        : {r['image']}")
        print(f"  Ground Truth : {r['ground_truth']}")
        print(f"  Prediction   : {r['prediction']}")
        print(f"  Raw Output   : {r['raw_model_output']}\n")

    await client.aclose()  # ✅ 释放连接池

# ===== 启动入口 =====
if __name__ == "__main__":
    asyncio.run(main())
    sys.exit(0)  # ✅ 强制退出，防止异步底层未回收导致挂起
