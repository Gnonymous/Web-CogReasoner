import os
import sys
import json
import base64
import asyncio
import aiofiles
from tqdm.asyncio import tqdm_asyncio
from openai import AsyncOpenAI
from rouge import Rouge

# CogReasoner / UI-TARs etc.
Test_Model = "Qwen2.5-VL-7B"  # 模型名称

# ===== 配置区 (已更新为 WebQA 任务) =====
# 请确保这个路径指向您为 WebQA 任务生成的 JSON 文件
TEST_JSON_PATH = "/code/CogReasoner/Test/VisualWebBench_webqa.json"
# 输出路径也已更新
OUTPUT_JSON_PATH = f"/code/CogReasoner/Code/Evalaute/Result/Test-{Test_Model}-VisualWebBench_WebQA.json"
MAX_SAMPLE = 245  # 测试样本上限 (根据您的数据集大小调整)
MAX_CONCURRENT_REQUESTS = 5  # 最大并发量
MODEL_NAME = "qwen2vl"  # 使用的大模型名称
BASE_URL = "http://localhost:8080/v1"  # vLLM兼容API地址

# ===== 初始化 openai 客户端 =====
client = AsyncOpenAI(
    api_key="EMPTY",
    base_url=BASE_URL,
)

# ===== 正式测评指标函数 (已替换为 eval_webqa) =====
def eval_webqa(preds, golds, **kwargs):
    """
    计算 WebQA 的 F1 分数。
    preds: 预测答案的列表。
    golds: 参考答案的列表的列表 (每个问题可以有多个参考答案)。
    """
    assert len(preds) == len(golds), "预测数量和参考答案数量必须一致"
    f1_scores = []
    # 注意：Rouge() 实例在循环外创建以提高效率
    rouge = Rouge(metrics=['rouge-1'])
    for pred, gold_list in zip(preds, golds):
        if not pred:
            pred = " "  # 避免空字符串导致ROUGE计算异常
        
        # 计算当前预测与所有参考答案的 F1 分数，并取最大值
        # gold_list 是当前问题的正确答案列表，例如 ['Sawfish'] 或 ['Answer A', 'Answer B']
        try:
            current_f1 = max([rouge.get_scores([pred], [gold], avg=True)['rouge-1']['f'] for gold in gold_list])
            f1_scores.append(current_f1)
        except Exception as e:
            # 如果发生错误（例如 gold_list 为空），则记录为0分并打印警告
            print(f"Warning: Could not compute F1 score for pred='{pred}' and gold_list='{gold_list}'. Error: {e}")
            f1_scores.append(0.0)

    # 确保 f1_scores 不为空，以避免除以零的错误
    if not f1_scores:
        return dict(f1=0.0)

    return dict(
        f1=sum(f1_scores) / len(f1_scores) * 100
    )

# ===== 单条样本推理函数 (已修改 ground_truth 的处理方式) =====
async def process_item(index, item, sem):
    async with sem:
        image_path = item["images"][0]
        
        # --- 关键修改 ---
        # `eval_webqa` 需要一个答案列表，所以我们将单个答案包装成列表
        # 即使只有一个正确答案，也需要是列表形式，例如 ['Sawfish']
        ground_truth = [item["messages"][1]["content"].strip()]
        
        user_prompt = item["messages"][0]["content"] # user_prompt 是包含 <image> 和问题的完整内容

        # 读取并编码图片
        async with aiofiles.open(image_path, "rb") as f:
            content = await f.read()
        encoded_image = base64.b64encode(content).decode("utf-8")
        image_data_uri = f"data:image;base64,{encoded_image}"

        try:
            # 从 user_prompt 中移除 <image> 标签，因为它不是模型输入的一部分
            prompt_text = user_prompt.replace("<image>\n", "").strip()

            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_data_uri}},
                            {"type": "text", "text": prompt_text},
                        ],
                    },
                ],
                temperature=0.1,
                top_p=0.95,
                max_tokens=1024,
            )
            pred_text = response.choices[0].message.content.strip()
        except Exception as e:
            pred_text = f"[ERROR] {str(e)}"

        return {
            "image": image_path,
            "ground_truth": ground_truth, # ground_truth 现在是一个列表
            "prediction": pred_text,
        }

# ===== 主函数 (已修改 metrics 的调用) =====
async def main():
    try:
        with open(TEST_JSON_PATH, "r", encoding="utf-8") as f:
            test_data = json.load(f)[:MAX_SAMPLE]
    except FileNotFoundError:
        print(f"错误：测试文件未找到，请检查路径: {TEST_JSON_PATH}")
        return

    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    tasks = [process_item(i, item, sem) for i, item in enumerate(test_data)]

    print(f"\n🚀 Starting evaluation for WebQA on {len(tasks)} samples...\n")
    results = await tqdm_asyncio.gather(*tasks)

    predictions = [r["prediction"] for r in results]
    references = [r["ground_truth"] for r in results] # 这现在是一个列表的列表

    # --- 关键修改 ---
    # 调用新的评估函数
    metrics = eval_webqa(predictions, references)

    output = {
        "task": "WebQA",
        "model": Test_Model,
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
    # 确保已安装 rouge-chinese 或 rouge
    try:
        from rouge import Rouge
    except ImportError:
        print("错误: rouge 库未安装。请运行 'pip install rouge' 或 'pip install rouge-chinese'")
        sys.exit(1)
        
    asyncio.run(main())
    sys.exit(0)