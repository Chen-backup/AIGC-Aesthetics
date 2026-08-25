import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import pandas as pd
import os
from tqdm import tqdm


def extract_dinov2_features():
    print("================ 1. 初始化 DINOv2 本地大模型 ================")
    # 自动使用 GPU 加速（如果有的话），否则使用 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前计算设备: {device}")

    # 🎯 本地模型路径 (保持不变)
    local_model_path = r"G:\E\CJH-SJTU\课题组\图像美学\code_3\dinov2-base-local"

    print(f"正在从本地目录极速加载模型: {local_model_path} ...")
    try:
        processor = AutoImageProcessor.from_pretrained(local_model_path)
        model = AutoModel.from_pretrained(local_model_path).to(device)
        model.eval()  # 设置为评估（推理）模式，不更新梯度
        print("✅ 本地模型加载成功！")
    except Exception as e:
        print(f"❌ 加载模型失败，请检查文件夹及文件。\n报错信息: {e}")
        return

    print("\n================ 2. 准备读取图片 ================")
    # 🎯 已经替换为您提供的真实图片绝对路径
    IMAGE_DIR = r"G:\E\CJH-SJTU\课题组\图像美学\Dataset\face_dataset"

    # 读取 mapping 表来获取图片名单 (确保在当前 code_3 目录下)
    mapping_path = "renumber&gender.xlsx"
    if not os.path.exists(mapping_path):
        print(f"❌ 找不到 {mapping_path}，请确保它和本代码在同一个目录下！")
        return

    df_mapping = pd.read_excel(mapping_path)
    image_names = df_mapping['face_id'].tolist()

    print(f"名单读取成功，共需要提取 {len(image_names)} 张图片的 DINOv2 特征。")

    print("\n================ 3. 开始提取绝对视觉黑盒特征 ================")
    features_list = []
    missing_images = []

    for img_name in tqdm(image_names, desc="特征提取进度"):
        img_path = os.path.join(IMAGE_DIR, img_name)

        if not os.path.exists(img_path):
            missing_images.append(img_name)
            continue

        try:
            # 读取图片并转换为 RGB
            image = Image.open(img_path).convert('RGB')

            # 图像预处理
            inputs = processor(images=image, return_tensors="pt").to(device)

            # 模型推理
            with torch.no_grad():
                outputs = model(**inputs)

            # 提取 CLS Token 的特征 (768维)
            img_emb = outputs.last_hidden_state[0, 0, :].cpu().numpy()

            # 将图片名和 768 维特征拼接，存入列表
            row_data = [img_name] + img_emb.tolist()
            features_list.append(row_data)

        except Exception as e:
            print(f"处理图片 {img_name} 时出错: {e}")

    if missing_images:
        print(f"\n⚠️ 警告：找不到以下 {len(missing_images)} 张图片，请检查图片文件夹：")
        print(missing_images[:5], "...")

    print("\n================ 4. 保存为 CSV 弹药库 ================")
    # 构造列名：image_name, emb_0, emb_1 ... emb_767
    col_names = ['image_name'] + [f'emb_{i}' for i in range(768)]
    df_dinov2 = pd.DataFrame(features_list, columns=col_names)

    save_name = 'dinov2_features.csv'
    df_dinov2.to_csv(save_name, index=False)
    print(f"🎉 提取完美收官！共 {len(df_dinov2)} 张图片的 768维 DINOv2 特征已保存至 '{save_name}'。")


if __name__ == "__main__":
    extract_dinov2_features()