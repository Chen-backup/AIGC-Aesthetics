import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # ⚠️ 新增：用于保存模型的库


def process_dinov2_pca_14():
    print("================ 1. 加载 DINOv2 深度特征 (768维) ================")
    # 读取刚刚提取好的 768 维特征文件
    df = pd.read_csv("dinov2_features.csv")
    emb_cols = [f'emb_{i}' for i in range(768)]  # DINOv2-base 是 768维

    # 清洗可能存在的缺失值 (防雷)
    initial_len = len(df)
    df = df.dropna(subset=emb_cols)
    print(f"✅ 成功加载 {len(df)} 张图片 (清除了 {initial_len - len(df)} 个无效行)，准备进入降维流水线。")

    print("\n================ 2. 构建标准化与 PCA 降维流水线 (Pipeline) ================")
    X = df[emb_cols].values

    # ⭐️ 核心修改：将 StandardScaler 和 PCA 打包成一个流水线
    # 这样它们就会共享同一个时空尺度，存成一个文件
    n_components = 14
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=n_components, random_state=42))  # 加入随机种子保证结果绝对可复现
    ])

    # 一键完成：先标准化，再做 PCA
    X_pca = pipeline.fit_transform(X)

    # 提取 pca 模块出来算解释方差
    pca_model = pipeline.named_steps['pca']
    explained_variance_ratio = pca_model.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)

    print(f"🎉 14维流水线降维完成！")
    print(f"👉 DINOv2 的前 14 个主成分累计解释了其 768 维空间中 【{cumulative_variance[-1]:.2%}】 的视觉信息。")
    print("💡 (请记下这个比例，它将写入你的论文中)")

    print("\n================ 3. 生成 DINOv2 降维诊断图表 ================")
    plt.figure(figsize=(10, 6))
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    plt.bar(range(1, n_components + 1), explained_variance_ratio, alpha=0.6, color='#ff7f0e',
            label='Individual PC Variance')
    plt.plot(range(1, n_components + 1), cumulative_variance, marker='o', linestyle='-', color='#d62728',
             label='Cumulative Variance')

    plt.title('DINOv2 Explained Variance: 768D to 14D (Fair Comparison)', fontsize=16, weight='bold')
    plt.xlabel('Principal Component (PC)', fontsize=14)
    plt.ylabel('Cumulative Explained Variance Ratio', fontsize=14)
    plt.xticks(range(1, n_components + 1, 1))
    plt.legend(loc='best')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('DINOv2_PCA_14_Variance.png', dpi=300)
    print("✅ 诊断图已保存为 'DINOv2_PCA_14_Variance.png'。")

    print("\n================ 4. 保存 DINOv2 贝叶斯输入表 & 降维模具 ================")
    # ⚠️ 核心修改：把列名直接改成大模型认识的 dino_dim_1 到 dino_dim_14
    df_pca = pd.DataFrame(X_pca, columns=[f'dino_dim_{i + 1}' for i in range(n_components)])

    # 把用来做关联匹配的 image_name 拼回去
    df_pca['image_name'] = df['image_name'].reset_index(drop=True)

    # 1. 保存 CSV 结果
    csv_save_path = 'dino_features_14d.csv'
    df_pca.to_csv(csv_save_path, index=False)
    print(f"✅ 纯净的 14 维 DINOv2 特征已保存为 '{csv_save_path}'。")

    # 2. ⭐️ 终极核心：把包含 Scaler 和 PCA 的联合流水线保存下来！
    # 如果你想把它存到 BYS_Fusion_28D_DINOv2_result 文件夹里，确保那个文件夹存在
    os.makedirs('BYS_Fusion_28D_DINOv2_result', exist_ok=True)
    pkl_save_path = r'BYS_Fusion_28D_DINOv2_result\dinov2_pca_14d.pkl'

    joblib.dump(pipeline, pkl_save_path)
    print(f"✅ 包含【标准化+PCA】的终极降维模具已保存为 '{pkl_save_path}'！！！")
    print("🚀 现在，你可以直接去运行 predict_new_photo.py 终极裁判打分系统了！")


if __name__ == "__main__":
    process_dinov2_pca_14()