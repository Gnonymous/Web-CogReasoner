import os
import sys
import json
import base64
import asyncio
import aiofiles
from tqdm.asyncio import tqdm_asyncio
from openai import AsyncOpenAI
from rouge import Rouge

# CogReasoner
# UI-TARs
Test_Model = "Qwen2.5-VL-7B"  # 模型名称

# ===== 配置区 =====
TEST_JSON_PATH = "/code/CogReasoner/Test/VisualWebBench_Heading_OCR_46.json"  # 测试数据路径
OUTPUT_JSON_PATH = f"/code/CogReasoner/Code/Evalaute/Result/Test-{Test_Model}-VisualWebBench_HeadingOCR_46.json"  # 推理结果保存路径
MAX_SAMPLE = 46  # 测试样本上限
MAX_CONCURRENT_REQUESTS = 5  # 最大并发量
ACCURACY_PRINT_INTERVAL = 10  # 每多少步打印一次中间结果
MODEL_NAME = "qwen2vl"  # 使用的大模型名称
BASE_URL = "http://localhost:8080/v1"  # vLLM兼容API地址

# ===== 初始化 openai 客户端 =====
client = AsyncOpenAI(
    api_key="EMPTY",
    base_url=BASE_URL,
)

# ===== 固定Prompt =====
FIXED_PROMPT = (
    "You are given a screenshot of a webpage. Please generate the main text within the screenshot, "
    "which can be regarded as the heading of the webpage.\n\n"
    "You should directly tell me the main content, and do not output any explanation or any other contents."
)

# ===== 正式测评指标函数 =====
def eval_heading_ocr(preds, golds, **kwargs):
    assert len(preds) == len(golds)
    for i in range(len(preds)):
        if not preds[i]:
            preds[i] = " "  # 避免ROUGE计算异常
    rouge = Rouge(metrics=['rouge-1', 'rouge-2', 'rouge-l'])
    scores = rouge.get_scores(preds, golds, avg=True)
    return dict(
        rouge_1=scores['rouge-1']['f'] * 100,
        rouge_2=scores['rouge-2']['f'] * 100,
        rouge_l=scores['rouge-l']['f'] * 100
    )

# ===== 单条样本推理函数 =====
async def process_item(index, item, sem):
    async with sem:
        image_path = item["images"][0]
        ground_truth = item["messages"][1]["content"].strip()

        # 读取并编码图片
        async with aiofiles.open(image_path, "rb") as f:
            content = await f.read()
        encoded_image = base64.b64encode(content).decode("utf-8")
        image_data_uri = f"data:image;base64,{encoded_image}"

        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_data_uri}},
                            {"type": "text", "text": FIXED_PROMPT},
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

        return {
            "image": image_path,
            "ground_truth": ground_truth,
            "prediction": pred_text,
        }

# ===== 主函数 =====
async def main():
    with open(TEST_JSON_PATH, "r", encoding="utf-8") as f:
        test_data = json.load(f)[:MAX_SAMPLE]

    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    tasks = [process_item(i, item, sem) for i, item in enumerate(test_data)]

    print(f"\n🚀 Starting evaluation on {len(tasks)} samples...\n")
    results = await tqdm_asyncio.gather(*tasks)

    predictions = [r["prediction"] for r in results]
    references = [r["ground_truth"] for r in results]

    metrics = eval_heading_ocr(predictions, references)

    output = {
        "metrics": metrics,
        "results": results,
    }

    # 保存结果
    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Evaluation Complete!")
    print(f"📊 Metrics: {json.dumps(metrics, indent=2)}")
    print(f"📁 Results saved at: {OUTPUT_JSON_PATH}")

    await client.close()

# ===== 启动入口 =====
if __name__ == "__main__":
    asyncio.run(main())
    sys.exit(0)