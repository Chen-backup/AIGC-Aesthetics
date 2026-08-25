import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


def process_dinov2_pca_14():
    print("================ 1. 加载 DINOv2 深度特征 (768维) ================")
    # 读取刚刚提取好的 768 维特征文件
    df = pd.read_csv("dinov2_features.csv")
    emb_cols = [f'emb_{i}' for i in range(768)]  # DINOv2-base 是 768维

    # 清洗可能存在的缺失值 (防雷)
    initial_len = len(df)
    df = df.dropna(subset=emb_cols)
    print(f"✅ 成功加载 {len(df)} 张图片 (清除了 {initial_len - len(df)} 个无效行)，准备进入标准化。")

    print("\n================ 2. 数据标准化 (Z-score) ================")
    X = df[emb_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("✅ 768维特征标准化完成。")

    print("\n================ 3. 执行 PCA 降维 (同等复杂度: 14维) ================")
    n_components = 14
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)

    print(f"🎉 14维 PCA 降维完成！")
    print(f"👉 DINOv2 的前 14 个主成分累计解释了其 768 维空间中 【{cumulative_variance[-1]:.2%}】 的视觉信息。")
    print("💡 (请记下这个比例，它将写入你的论文中)")

    print("\n================ 4. 生成 DINOv2 降维诊断图表 ================")
    plt.figure(figsize=(10, 6))
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    # 画图：使用了代表 DINOv2 差异化的橙色柱子
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

    print("\n================ 5. 保存 DINOv2 的贝叶斯输入表 ================")
    # 构造新的特征表，包含 PC1, PC2 ... PC14
    df_pca = pd.DataFrame(X_pca, columns=[f'PC{i + 1}' for i in range(n_components)])

    # 把用来做关联匹配的 image_name 拼回去
    df_pca['image_name'] = df['image_name'].reset_index(drop=True)

    # 保存为专属的 DINOv2 CSV 文件，避免和 InsightFace 的覆盖
    df_pca.to_csv('PCA_14_dinov2.csv', index=False)
    print("✅ 纯净的 14 维 DINOv2 特征已保存为 'PCA_14_dinov2.csv'。")


if __name__ == "__main__":
    process_dinov2_pca_14()