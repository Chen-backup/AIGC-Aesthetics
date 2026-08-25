import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


def process_stylegan_pca_14():
    print("================ 1. 加载 StyleGAN-W+ 深度生成特征 (9216维) ================")
    # 读取刚刚提取好的 9216 维特征文件
    df = pd.read_csv("stylegan_w_features.csv")

    # 构造 StyleGAN 的 9216 维特征列名
    w_cols = [f'w_{i}' for i in range(9216)]

    # 清洗可能存在的缺失值 (防雷)
    initial_len = len(df)
    df = df.dropna(subset=w_cols)
    print(f"✅ 成功加载 {len(df)} 张图片 (清除了 {initial_len - len(df)} 个无效行)，准备进入标准化。")

    print("\n================ 2. 数据标准化 (Z-score) ================")
    # 注意：9216 维数据标准化对内存有一点点要求，但 300 张图瞬间就能算完
    X = df[w_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("✅ 9216维高维生成流形标准化完成。")

    print("\n================ 3. 执行 PCA 降维 (同等复杂度: 14维) ================")
    n_components = 14
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)

    print(f"🎉 14维 PCA 降维完成！")
    print(f"👉 StyleGAN 的前 14 个主成分累计解释了其 9216 维生成流形空间中 【{cumulative_variance[-1]:.2%}】 的视觉变异。")
    print("💡 (请记下这个强悍的比例，它是你论文中证明降维合理性的核心论据)")

    print("\n================ 4. 生成 StyleGAN 降维诊断图表 ================")
    plt.figure(figsize=(10, 6))
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    # 画图：使用了代表 StyleGAN 潜空间的科技蓝紫配色
    plt.bar(range(1, n_components + 1), explained_variance_ratio, alpha=0.7, color='#9b59b6',
            label='Individual PC Variance')
    plt.plot(range(1, n_components + 1), cumulative_variance, marker='D', linestyle='-', color='#2980b9',
             label='Cumulative Variance', linewidth=2)

    plt.title('StyleGAN-W+ Explained Variance: 9216D to 14D (Fair Comparison)', fontsize=16, weight='bold')
    plt.xlabel('Principal Component (PC)', fontsize=14)
    plt.ylabel('Cumulative Explained Variance Ratio', fontsize=14)
    plt.xticks(range(1, n_components + 1, 1))

    # 标注出累计方差的最终值
    plt.text(n_components, cumulative_variance[-1] + 0.02, f'{cumulative_variance[-1]:.1%}',
             ha='center', va='bottom', weight='bold', color='#2980b9')

    plt.legend(loc='best')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('StyleGAN_PCA_14_Variance.png', dpi=300)
    print("✅ 诊断图已保存为 'StyleGAN_PCA_14_Variance.png'。")

    print("\n================ 5. 保存 StyleGAN 的贝叶斯输入表 ================")
    # 构造新的特征表，包含 PC1, PC2 ... PC14
    df_pca = pd.DataFrame(X_pca, columns=[f'PC{i + 1}' for i in range(n_components)])

    # 把用来做关联匹配的 image_name 拼回去
    df_pca['image_name'] = df['image_name'].reset_index(drop=True)

    # 保存为专属的 StyleGAN CSV 文件
    df_pca.to_csv('PCA_14_stylegan_w.csv', index=False)
    print("✅ 纯净的 14 维 StyleGAN 特征已保存为 'PCA_14_stylegan_w.csv'。")


if __name__ == "__main__":
    process_stylegan_pca_14()