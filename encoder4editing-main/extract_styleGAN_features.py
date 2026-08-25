import torch
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from argparse import Namespace

# 导入 e4e 核心库 (无需再导入 dlib 和 alignment)
try:
    from models.psp import pSp
except ImportError:
    print("❌ 找不到 e4e 模型库！请确保本代码保存在 encoder4editing-main 文件夹内运行。")
    exit()


def extract_stylegan_w_features_direct():
    print("================ 1. 初始化 StyleGAN (e4e) 本地大模型 ================")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前计算设备: {device}")

    # 🎯 预训练模型路径
    local_model_path = "e4e_ffhq_encode.pt"

    if not os.path.exists(local_model_path):
        print(f"❌ 找不到权重文件 {local_model_path}，请确认下载位置！")
        return

    print(f"正在从本地极速加载 e4e 权重: {local_model_path} ...")
    try:
        ckpt = torch.load(local_model_path, map_location='cpu')
        opts = ckpt['opts']
        opts['checkpoint_path'] = local_model_path
        opts['device'] = device.type
        opts = Namespace(**opts)

        net = pSp(opts)
        net.eval()
        net.to(device)
        print("✅ e4e 模型加载成功！(已彻底移除 dlib 依赖)")

    except Exception as e:
        print(f"❌ 加载模型失败，报错信息: {e}")
        return

    print("\n================ 2. 准备读取图片 ================")
    # 🎯 你的真实图片绝对路径
    IMAGE_DIR = r"G:\E\CJH-SJTU\课题组\图像美学\Dataset\face_dataset"
    mapping_path = r"G:\E\CJH-SJTU\课题组\图像美学\code_3\renumber&gender.xlsx"

    if not os.path.exists(mapping_path):
        print(f"❌ 找不到 {mapping_path}！")
        return

    df_mapping = pd.read_excel(mapping_path)
    image_names = df_mapping['face_id'].tolist()

    print(f"名单读取成功，共需要提取 {len(image_names)} 张 AI 合成图片的特征。")

    print("\n================ 3. 开始提取绝对生成流形特征 ================")
    features_list = []
    missing_images = []
    failed_images = []

    # 图像预处理流水线：直接转 256x256 并归一化，拒绝二次对齐！
    img_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    for img_name in tqdm(image_names, desc="潜空间映射进度"):
        img_path = os.path.join(IMAGE_DIR, img_name)

        if not os.path.exists(img_path):
            missing_images.append(img_name)
            continue

        try:
            # 1. 直接读取并强制转为 RGB（消除透明通道干扰）
            raw_image = Image.open(img_path).convert('RGB')

            # 2. 直接放缩并进入 GPU
            input_tensor = img_transforms(raw_image).unsqueeze(0).to(device)

            # 3. e4e 模型推理，提取 W+ 向量
            with torch.no_grad():
                # randomize_noise=False 保证特征稳定，return_latents=True 只索取 18x512 向量
                _, latents = net(input_tensor, randomize_noise=False, return_latents=True)

            # 4. 展平成 9216 维的一维数组
            latent_flat = latents.cpu().numpy().flatten()

            row_data = [img_name] + latent_flat.tolist()
            features_list.append(row_data)

        except Exception as e:
            # 🚨 如果再报错，这次会打印出真正的底层错误原因！
            print(f"\n❌ 图片 {img_name} 爆出的真实错误是: {e}")
            failed_images.append(img_name)
            # 遇到严重错误直接暂停，方便调试
            break

            # 异常报告
    if missing_images:
        print(f"\n⚠️ 警告：找不到 {len(missing_images)} 张图片。")
    if failed_images:
        print(f"\n⚠️ 警告：有 {len(failed_images)} 张图片提取失败。")

    print("\n================ 4. 保存为 CSV 弹药库 ================")
    if not features_list:
        print("❌ 提取失败：没有成功处理任何图片。")
        return

    # 构造 9216 维度的列名：image_name, w_0, w_1 ... w_9215
    col_names = ['image_name'] + [f'w_{i}' for i in range(9216)]
    df_stylegan = pd.DataFrame(features_list, columns=col_names)

    # 统一保存到 code_3 文件夹
    save_path = r"G:\E\CJH-SJTU\课题组\图像美学\code_3\stylegan_w_features.csv"
    df_stylegan.to_csv(save_path, index=False)

    print(f"🎉 提取完美收官！共 {len(df_stylegan)} 张图片的 9216 维 W+ 潜向量已保存至:\n'{save_path}'")


if __name__ == "__main__":
    extract_stylegan_w_features_direct()