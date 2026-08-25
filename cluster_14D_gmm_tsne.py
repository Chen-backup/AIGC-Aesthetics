import arviz as az
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE

print("\n" + "★" * 60)
print("🌌 启动【14维审美基因：GMM软聚类 + t-SNE非线性降维分析】")
print("★" * 60)

# ==========================================
# 1. 全局配置与中文字体修复
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

DIR_NC = "BYS_Ultimate_Heterogeneity_Results"
SAVE_DIR = "BYS_Clustering_Results_Advanced"
os.makedirs(SAVE_DIR, exist_ok=True)

# 14 个可解释特征
TARGET_FEATURES = [
    'face_hw_ratio', 'eye_face_w_ratio', 'mouth_face_w_ratio', 'three_courts_balance',
    'upper_lower_ratio', 'eye_y_ratio', 'total_symmetry', 'le_nose_re_angle',
    'mouth_nose_ratio', 'face_brightness', 'face_contrast', 'face_clarity',
    'saturation', 'edge_density'
]

# ==========================================
# 2. 提取 14 维度偏好数据
# ==========================================
print("\n⏳ 正在潜入 14 个时间胶囊，提取 1300 位评委的全维度潜意识偏好...")
rater_slopes_list = []

for feat in TARGET_FEATURES:
    nc_path = os.path.join(DIR_NC, f"Ultimate_Heterogeneity_{feat}.nc")
    if not os.path.exists(nc_path):
        print(f"❌ 找不到文件 {nc_path}，跳过该特征！")
        continue

    trace = az.from_netcdf(nc_path)
    var_name = f"{feat}|rater"
    try:
        mean_slopes = trace.posterior[var_name].mean(dim=["chain", "draw"])
        s = mean_slopes.to_series()
        s.name = feat
        rater_slopes_list.append(s)
    except KeyError:
        pass
    del trace

df_preferences = pd.concat(rater_slopes_list, axis=1).dropna()
print(f"✅ 成功构建人群偏好矩阵！共有 {len(df_preferences)} 名被试，{df_preferences.shape[1]} 个维度。")

# ==========================================
# 3. 数据标准化 (Z-score) 与 GMM 聚类
# ==========================================
print("\n🧬 正在执行 Z-score 标准化与 Gaussian Mixture Model (GMM) 聚类...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_preferences)

# 使用 BIC (贝叶斯信息准则) 自动寻找最优的流派数量 (通常 3-6)
# 这里为了确保结果的业务可解释性，我们强制或推荐使用 4 类
K_CLUSTERS = 2

gmm = GaussianMixture(n_components=K_CLUSTERS, covariance_type='full', random_state=42)
cluster_labels = gmm.fit_predict(X_scaled)
df_preferences['Cluster'] = cluster_labels

# ==========================================
# 4. t-SNE 非线性降维
# ==========================================
print("🌌 正在启动 t-SNE 流形降维引擎（将 14 维空间压缩至 2 维）...")
# perplexity 是 t-SNE 的核心参数，通常在 30-50 之间，决定了关注局部还是全局
tsne = TSNE(n_components=2, perplexity=40, random_state=42)
X_2d = tsne.fit_transform(X_scaled)

df_preferences['tSNE_1'] = X_2d[:, 0]
df_preferences['tSNE_2'] = X_2d[:, 1]

# ==========================================
# 5. 可视化一：t-SNE 人群星空散点图
# ==========================================
print("\n🎨 正在渲染 t-SNE 人群流派 2D 散点星空图...")
plt.figure(figsize=(10, 8))

# 使用高级的 seaborn 散点图
sns.scatterplot(
    x='tSNE_1', y='tSNE_2',
    hue='Cluster',
    palette='Set1',  # 强对比色
    data=df_preferences,
    legend='full',
    alpha=0.7,
    edgecolor='w',
    s=60
)

plt.title('t-SNE Visualization of Aesthetic Preference Clusters (14D → 2D)', fontsize=16, weight='bold')
plt.xlabel('t-SNE Dimension 1', fontsize=12)
plt.ylabel('t-SNE Dimension 2', fontsize=12)
plt.legend(title='Aesthetic Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

tsne_path = os.path.join(SAVE_DIR, "tSNE_Audience_Clusters_2D.png")
plt.savefig(tsne_path, dpi=300)

# ==========================================
# 6. 可视化二：GMM 聚类中心热力图 (解释流派含义)
# ==========================================
print("🎨 正在渲染 GMM 流派偏好解析热力图...")
# GMM 的 cluster_centers_ 即为其每个高斯分布的均值 (means_)
gmm_centers = pd.DataFrame(gmm.means_, columns=df_preferences.columns[:-3])  # 排除 cluster 和 tsne 列

plt.figure(figsize=(12, 9))
sns.heatmap(gmm_centers.T,
            cmap="RdBu_r",
            center=0,
            annot=True,
            fmt=".3f",
            cbar_kws={'label': 'Z-scored Preference Strength'},
            annot_kws={"size": 11, "weight": "bold"},
            linewidths=1.5,
            linecolor='white')

plt.title(f'GMM Cluster Centers (Aesthetic Profiles)', fontsize=18, weight='bold', pad=20)
plt.xlabel('Gaussian Mixture Clusters (审美流派)', fontsize=14, weight='bold')
plt.ylabel('14 Interpretable Features', fontsize=14, weight='bold')

# 统计人数并在 X 轴显示
cluster_counts = df_preferences['Cluster'].value_counts().sort_index()
xtick_labels = [f"Cluster {i}\n(N={cluster_counts[i]})" for i in range(K_CLUSTERS)]
plt.xticks(ticks=np.arange(K_CLUSTERS) + 0.5, labels=xtick_labels, fontsize=12)
plt.yticks(rotation=0, fontsize=12)

plt.tight_layout()

heatmap_path = os.path.join(SAVE_DIR, "GMM_Cluster_Profiles_Heatmap.png")
plt.savefig(heatmap_path, dpi=300)

print(f"\n✅ 史诗级洞察完成！")
print(f"👉 1. t-SNE 降维散点图已保存: {tsne_path}")
print(f"👉 2. GMM 画像热力图已保存: {heatmap_path}")