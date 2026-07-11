import os
import json
import base64
import re
import asyncio
import aiofiles
from pathlib import Path
from tqdm.asyncio import tqdm_asyncio
from openai import AsyncOpenAI
from rouge import Rouge

Model_name = "Web-CogReasoner"
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[3])).resolve()

def resolve_asset_path(path):
    path = Path(path)
    if path.is_absolute():
        try:
            path = path.relative_to("/code/Web-CogReasoner")
        except ValueError:
            return str(path)
    return str(PROJECT_ROOT / path)

# Config
TEST_JSON_PATH = str(PROJECT_ROOT / "benchmark/Memorizing/Element_Attribute_249.json")
MODEL_NAME = "qwen2vl"
MAX_SAMPLE = 249
MAX_CONCURRENT_REQUESTS = 5
ACCURACY_PRINT_INTERVAL = 10
OUTPUT_JSON_PATH = str(PROJECT_ROOT / f"results_Web-CogBench/{Model_name}-Element-Attribute.json")

# OpenAI client and ROUGE
client = AsyncOpenAI(
    api_key=os.getenv("MODEL_API_KEY", "EMPTY"),
    base_url=os.getenv("MODEL_ENDPOINT", "http://localhost:8080/v1"),
)
rouge = Rouge(metrics=['rouge-1'])

def parse_model_output(text):
    if text is None:
        text = ""
    role_match = re.search(r"Role:\s*\[?([^\]\n,]+)\]?", text, re.IGNORECASE)
    name_match = re.search(r"Name:\s*\[?([^\]\n,]+)\]?", text, re.IGNORECASE)
    
    role = role_match.group(1).strip() if role_match else None
    name = name_match.group(1).strip() if name_match else None
    
    return role, name

async def process_item(index, item, sem, stats):
    async with sem:
        image_path = resolve_asset_path(item["images"][0])
        gt_response = item["messages"][-1]["content"]

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
                            {
                                "type": "text",
                                "text": (
                                        '''
                                        You are viewing a screenshot of a webpage where a specific element is marked with a red box. \nYour task is to predict its ARIA role and accessible name based on the context of the webpage and the visual appearance of the element. Please make your prediction using the following list of roles with their semantic descriptions, combined with visual clues from the screenshot (such as text, position, and style).\n** Possible Roles and Their Semantics\n1.link: A hyperlink used to navigate to other pages or resources.\n2.button: A button used to trigger actions (e.g., submit, confirm).\n3.textbox: A single-line text input field for entering free text.\n4.searchbox: A search input field for entering search queries.\n5.checkbox: A checkbox for multiple-choice options.\n6.radio: A radio button for single-choice options.\n7.slider: A slider for adjusting a range of values.\n8.spinbutton: A numeric adjuster for incrementing or decrementing values.\n9.combobox: A dropdown selection box allowing choices from options.\n10.option: A single option, typically within a dropdown or list.\n11.listbox: A list selection box displaying multiple selectable items.\n12.img: An image used to display visual content.\n13.form: A form containing a collection of user input controls.\n14.navigation: A navigation area providing links for the page or site.\n16.banner: A header, typically containing the site title or banner.\n17.contentinfo: A footer, usually containing copyright or contact information.\n18.article: An article, an independent content block (e.g., news, post).\n19.search: A search area, typically containing search functionality.\n20.heading: A heading used for content hierarchy (level may need to be inferred, e.g., heading level 1).\n21.list: A list containing multiple items.\n22.listitem: A list item, a single entry within a list.\n23.table: A table for displaying data in rows and columns.\n24.row: A table row containing cells.\n25.columnheader: A column header in a table.\n26.rowheader: A row header in a table.\n27.cell: A cell, a data item in a table.\n28.dialog: A dialog box, such as a popup window or modal.\n29.progressbar: A progress bar showing task progress.\n30.status: A status update providing dynamic information.\n31.paragraph: A paragraph, a block of text content.\n\n** Prediction Guidance\n1. Role:\nSelect the most matching role based on the element\u2019s visual characteristics (e.g., button shape, input field border) and context (e.g., located in a navigation bar or form).\nRefer to the role semantics to ensure the prediction aligns with its definition.\nIf uncertain, prioritize values from the above role list and avoid arbitrary guesses.\n2. Name:\nExtract the name from visible text on the element (e.g., \u201cSubmit\u201d on a button, a label next to an input field).\nIf no visible text is present, infer a reasonable name (e.g., \u201cunlabeled button\u201d).\n** Output Format\nPlease provide your prediction in the following format:\nRole: [role], Name: [name]\n
                                        '''
                                ),
                            },
                        ],
                    },
                ],
                temperature=0.1,
                top_p=0.95,
                max_tokens=2048,
            )
            pred_text = response.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"Model inference failed for sample {index}: {e}") from e

        pred_role, pred_name = parse_model_output(pred_text)
        gt_role, gt_name = parse_model_output(gt_response)
        
        match_role = pred_role == gt_role

        pred_for_rouge = pred_name if pred_name else " "
        gt_for_rouge = gt_name if gt_name else " "
        
        try:
            scores = rouge.get_scores([pred_for_rouge], [gt_for_rouge], avg=True)
            name_f1 = scores['rouge-1']['f']
            name_precision = scores['rouge-1']['p']
            name_recall = scores['rouge-1']['r']
        except Exception:
            name_f1, name_precision, name_recall = 0.0, 0.0, 0.0

        match_name = name_f1 == 1.0

        match_all = match_role and match_name

        stats["total"] += 1
        stats["role_correct"] += int(match_role)
        stats["all_correct"] += int(match_all)
        stats["name_f1_total"] += name_f1
        stats["name_precision_total"] += name_precision
        stats["name_recall_total"] += name_recall

        if stats["total"] % ACCURACY_PRINT_INTERVAL == 0:
            role_acc = stats["role_correct"] / stats["total"] * 100
            avg_name_f1 = stats["name_f1_total"] / stats["total"] * 100
            full_acc = stats["all_correct"] / stats["total"] * 100
            print(f"\nStep {stats['total']}: Role Acc={role_acc:.2f}%, Avg Name ROUGE-1 F1={avg_name_f1:.2f}%, Full Score={(role_acc + avg_name_f1) / 2:.2f}%\n")

        return {
            "image": os.path.basename(image_path),
            "ground_truth": {"role": gt_role, "name": gt_name},
            "prediction": {"role": pred_role, "name": pred_name},
            "metrics_per_sample": {
                "name_rouge1_f1": name_f1,
                "name_rouge1_precision": name_precision,
                "name_rouge1_recall": name_recall
            },
            "match_role": match_role,
            "match_name": match_name,
            "match_all": match_all
        }

async def main():
    with open(TEST_JSON_PATH, "r", encoding="utf-8") as f:
        test_data = json.load(f)[:MAX_SAMPLE]

    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    stats = {
        "total": 0, 
        "role_correct": 0, 
        "all_correct": 0,
        "name_f1_total": 0.0,
        "name_precision_total": 0.0,
        "name_recall_total": 0.0
    }
    tasks = [process_item(i, item, sem, stats) for i, item in enumerate(test_data)]

    print(f"\n🚀 Starting async evaluation of {len(tasks)} samples...\n")
    results = await tqdm_asyncio.gather(*tasks)

    total_samples = stats["total"] if stats["total"] > 0 else 1
    
    role_acc = (stats["role_correct"] / total_samples * 100)
    full_acc = (stats["all_correct"] / total_samples * 100)
    
    avg_name_precision = (stats["name_precision_total"] / total_samples * 100)
    avg_name_recall = (stats["name_recall_total"] / total_samples * 100)
    avg_name_f1 = (stats["name_f1_total"] / total_samples * 100)

    errors = [r for r in results if not r["match_all"]]

    output = {
        "metrics": {
            "total_samples": stats["total"],
            "role_accuracy": role_acc,
            "name_metrics": {
                "average_rouge1_precision": avg_name_precision,
                "average_rouge1_recall": avg_name_recall,
                "average_rouge1_f1_score": avg_name_f1,
            },
            "full_match_accuracy": full_acc,
            "full_score":(role_acc + avg_name_f1) / 2
        },
        "errors": errors
    }

    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nEvaluation Complete")
    print(f"Role Accuracy             : {role_acc:.2f}%")
    print(f"Avg Name ROUGE-1 Precision: {avg_name_precision:.2f}%")
    print(f"Avg Name ROUGE-1 Recall   : {avg_name_recall:.2f}%")
    print(f"Avg Name ROUGE-1 F1-Score : {avg_name_f1:.2f}%")
    print(f"Full Match Accuracy       : {full_acc:.2f}%")
    print(f"Full Score                : {(role_acc + avg_name_f1) / 2:.2f}%")
    print(f"Results saved to: {OUTPUT_JSON_PATH}")

    print("\n❌ Sample Errors (up to 5):")
    for r in errors[:5]:
        print(f"- Image        : {r['image']}")
        print(f"  Ground Truth : {r['ground_truth']}")
        print(f"  Prediction   : {r['prediction']}")
        print(f"  Name ROUGE-1 F1: {r['metrics_per_sample']['name_rouge1_f1']:.2f}\n")

if __name__ == "__main__":
    asyncio.run(main())
