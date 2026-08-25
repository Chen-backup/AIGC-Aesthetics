import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


def process_insightface_pca_14():
    print("================ 1. 加载 InsightFace 深度特征 ================")
    file_path = "face_features.csv"
    print(f"正在读取 {file_path}，请稍候...")
    df = pd.read_csv(file_path)
    emb_cols = [f'emb_{i}' for i in range(512)]

    print("\n================ 1.5. 清洗缺失值 (防雷行动) ================")
    df = df.dropna(subset=emb_cols)
    print(f"✅ 现存 {len(df)} 张干净图片，准备进入标准化。")

    print("\n================ 2. 数据标准化 (Z-score) ================")
    X = df[emb_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\n================ 3. 执行 PCA 降维 (公平对决: 14维) ================")
    # 🎯 核心修改：严格提取 14 个主成分
    n_components = 14
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)

    print(f"🎉 14维 PCA 降维完成！")
    print(f"👉 这 14 个主成分 (PC) 累计解释了原始 512 维特征中 【{cumulative_variance[-1]:.2%}】 的信息方差。")
    print("💡 (请记下这个方差比例，它可以写进论文里说明 14 维保留了多少 AI 视觉信息)")

    print("\n================ 4. 生成降维诊断图表 ================")
    plt.figure(figsize=(10, 6))
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    plt.bar(range(1, n_components + 1), explained_variance_ratio, alpha=0.6, color='#1f77b4',
            label='Individual PC Variance')
    plt.plot(range(1, n_components + 1), cumulative_variance, marker='o', linestyle='-', color='#d62728',
             label='Cumulative Variance')

    plt.title('PCA Explained Variance: 512D to 14D (Fair Comparison)', fontsize=16, weight='bold')
    plt.xlabel('Principal Component (PC)', fontsize=14)
    plt.ylabel('Cumulative Explained Variance Ratio', fontsize=14)
    plt.xticks(range(1, n_components + 1, 1))
    plt.legend(loc='best')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('InsightFace_PCA_14_Variance.png', dpi=300)

    print("\n================ 5. 保存 14 维特征表 ================")
    df_pca = pd.DataFrame(X_pca, columns=[f'PC{i + 1}' for i in range(n_components)])
    df_pca['image_name'] = df['image_name'].reset_index(drop=True)
    df_pca.to_csv('PCA_14_features.csv', index=False)
    print("✅ 纯净的 14 维主成分特征已保存为 'PCA_14_features.csv'。")


if __name__ == "__main__":
    process_insightface_pca_14()