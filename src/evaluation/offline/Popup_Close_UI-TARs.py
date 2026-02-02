import json
import os
import re
from PIL import Image, ImageDraw

def parse_click_coordinates(raw_output: str) -> tuple[int, int] | None:
    """
    从模型的原始输出字符串中解析出点击的 (x, y) 坐标。
    """
    match = re.search(r'\((\d+),\s*(\d+)\)', raw_output)
    if match:
        x = int(match.group(1))
        y = int(match.group(2))
        return (x, y)
    print(f"⚠️  警告: 无法从 '{raw_output}' 中解析坐标。")
    return None

def annotate_image(
    image_path: str,
    coords: tuple[int, int],
    output_path: str,
    box_size: int = 50,         # 增大方框尺寸
    crosshair_size: int = 25,   # 十字准星的半径
    line_width: int = 5,        # 增粗线条
    color: str = "red"
):
    """
    在图片的指定坐标上绘制一个带十字准星的醒目红框。

    Args:
        image_path (str): 原始图片的路径。
        coords (tuple[int, int]): 要标注的 (x, y) 坐标。
        output_path (str): 保存标注后图片的路径。
        box_size (int): 标注框的边长。
        crosshair_size (int): 十字准星从中心点向外延伸的长度。
        line_width (int): 标注线条的宽度。
        color (str): 标注的颜色。
    """
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGBA")
            draw = ImageDraw.Draw(img)
            x, y = coords

            # --- 绘制外部方框 ---
            half_box = box_size // 2
            box_coords = [
                (x - half_box, y - half_box),  # 左上角
                (x + half_box, y + half_box)   # 右下角
            ]
            draw.rectangle(box_coords, outline=color, width=line_width)

            # --- 绘制中心十字准星 ---
            # 水平线
            draw.line([(x - crosshair_size, y), (x + crosshair_size, y)], fill=color, width=line_width - 1)
            # 垂直线
            draw.line([(x, y - crosshair_size), (x, y + crosshair_size)], fill=color, width=line_width - 1)

            img.save(output_path)
            print(f"✅ 已标注并保存到: {output_path}")

    except FileNotFoundError:
        print(f"❌ 错误: 找不到原始图片 '{image_path}'。")
    except Exception as e:
        print(f"❌ 处理图片 '{image_path}' 时发生错误: {e}")

def main():
    """
    主函数，读取JSON，处理并生成标注图片。
    """
    input_json_file = '/code/CogReasoner/Code/Evalaute/Result/Test-UI-TARs-Single_Step.json'  # 你的JSON文件名
    output_dir = '/code/CogReasoner/Code/Evalaute/Result/Single_Step_UI-TARs'        # 输出文件夹名

    os.makedirs(output_dir, exist_ok=True)

    try:
        with open(input_json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ 错误: 找不到JSON文件 '{input_json_file}'。")
        return
    except json.JSONDecodeError:
        print(f"❌ 错误: JSON文件 '{input_json_file}' 格式不正确。")
        return

    for i, result in enumerate(data.get('results', [])):
        print(f"\n--- 正在处理第 {i+1} 个结果 ---")

        raw_output = result.get('raw_model_output')
        image_list = result.get('images')

        if not raw_output or not image_list:
            print("⚠️  警告: 此条目缺少 'raw_model_output' 或 'images'，已跳过。")
            continue

        coords = parse_click_coordinates(raw_output)
        if not coords:
            continue

        original_image_path = image_list[0]
        base_name = os.path.basename(original_image_path)
        name, ext = os.path.splitext(base_name)
        new_filename = f"{i+1:02d}_{name}_annotated{ext}"
        output_image_path = os.path.join(output_dir, new_filename)

        # 调用增强版的标注函数
        annotate_image(original_image_path, coords, output_image_path)

if __name__ == "__main__":
    main()