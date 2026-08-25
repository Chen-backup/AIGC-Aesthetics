import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import os

print("\n" + "★" * 60)
print("🌳 启动【全自动寻优版：高争议特征层次聚类分析】")
print("★" * 60)

# ==========================================
# 1. 基础配置与数据加载
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

SAVE_DIR = "BYS_Clustering_Results_Advanced"
os.makedirs(SAVE_DIR, exist_ok=True)

CSV_PATH = os.path.join(SAVE_DIR, "Rater_14D_Preferences.csv")

try:
    df_preferences = pd.read_csv(CSV_PATH, index_col=0)
    features_only = [col for col in df_preferences.columns if col not in ['Cluster', 'tSNE_1', 'tSNE_2']]
    df_raw = df_preferences[features_only]
except FileNotFoundError:
    print(f"❌ 找不到文件 {CSV_PATH}！")
    exit()

# ==========================================
# 2. 提取“高争议特征” (Top Variance)
# ==========================================
print("\n🔍 正在剥离‘人类共识’，提取高争议核心特征...")
variances = df_raw.var().sort_values(ascending=False)
TOP_N = 6
controversial_features = variances.head(TOP_N).index.tolist()

X_controversial = df_raw[controversial_features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_controversial)

# ==========================================
# 3. 数学寻优：自动寻找最佳分割层级 (K)
# ==========================================
print("\n📐 正在通过数学指标扫描树状图，寻找最佳剪枝点...")
Z = linkage(X_scaled, method='ward')

k_range = range(2, 11)
silhouette_scores = []
ch_scores = []

for k in k_range:
    # 按照当前的 K 值对树进行切分
    labels = fcluster(Z, t=k, criterion='maxclust')
    silhouette_scores.append(silhouette_score(X_scaled, labels))
    ch_scores.append(calinski_harabasz_score(X_scaled, labels))

# 打印详细指标供写论文用
print("\n📊 【层次聚类各 K 值评估指标】")
print("-" * 55)
print(f"{'K (聚类数)':<10} | {'Silhouette (越高越好)':<20} | {'CH Index (越高越好)'}")
print("-" * 55)
for i, k in enumerate(k_range):
    print(f"K = {k:<6} | {silhouette_scores[i]:<20.4f} | {ch_scores[i]:.2f}")
print("-" * 55)

# 根据 Silhouette 找出最优 K
optimal_k_sil = k_range[np.argmax(silhouette_scores)]
# 根据 CH Index 找出最优 K
optimal_k_ch = k_range[np.argmax(ch_scores)]

print(f"\n🏆 算法诊断结果：")
print(f"   - 轮廓系数推荐最佳分类数：K = {optimal_k_sil}")
print(f"   - CH 指数推荐最佳分类数：K = {optimal_k_ch}")

# 决定最终 K 值（如果两者不同，默认采纳轮廓系数，或可以修改为取平均/更合理的那个）
FINAL_K = optimal_k_sil
print(f"🎯 自动锁定终极聚类数目：将树状图精准切分为 【{FINAL_K} 大流派】！")

# ==========================================
# 4. 绘制带有自动辅助线的树状图
# ==========================================
print("\n🌳 正在渲染带切线的审美流派进化树...")
plt.figure(figsize=(14, 7))
plt.title(f'Hierarchical Dendrogram of Aesthetic Factions (Auto-Cut at K={FINAL_K})', fontsize=16, weight='bold')
plt.xlabel('Individual Raters', fontsize=12)
plt.ylabel('Aesthetic Distance (Ward)', fontsize=12)

# 动态计算切线的确切高度 (Y轴坐标)
# Z矩阵的倒数第 FINAL_K-1 行的第三个元素（距离）就是我们要切下去的高度
cut_distance = Z[-(FINAL_K-1), 2] if FINAL_K > 1 else Z[-1, 2]

dendro = dendrogram(
    Z,
    truncate_mode='level',
    p=5,
    leaf_rotation=90.,
    leaf_font_size=10.,
    show_contracted=True,
    color_threshold=cut_distance # 自动按最佳距离上色
)

# 画出自动算出来的最佳切线
plt.axhline(y=cut_distance, color='r', linestyle='--', linewidth=2,
            label=f'Optimal Cut (Height={cut_distance:.1f}, K={FINAL_K})')
plt.legend()
plt.tight_layout()

tree_path = os.path.join(SAVE_DIR, "Aesthetic_Evolution_Tree_AutoCut.png")
plt.savefig(tree_path, dpi=300)

# ==========================================
# 5. 按照最优 K 生成最终画像热力图
# ==========================================
print(f"🎨 正在生成 {FINAL_K} 大流派画像热力图...")
df_raw['Final_Cluster'] = fcluster(Z, t=FINAL_K, criterion='maxclust')

cluster_centers = df_raw.groupby('Final_Cluster')[controversial_features].mean()

plt.figure(figsize=(10, 6))
sns.heatmap(cluster_centers.T, cmap="RdBu_r", center=0, annot=True, fmt=".3f",
            cbar_kws={'label': 'Preference Strength'}, annot_kws={"size": 12, "weight": "bold"},
            linewidths=1.5, linecolor='white')

plt.title(f'Profiles of {FINAL_K} Auto-Detected Factions (Top {TOP_N} Divisive Features)', fontsize=16, weight='bold')
plt.xlabel('Hierarchical Factions (最优流派)', fontsize=12)
plt.tight_layout()

heat_path = os.path.join(SAVE_DIR, "Auto_Divisive_Factions_Heatmap.png")
plt.savefig(heat_path, dpi=300)

print(f"\n✅ 全部完成！")
print(f"👉 树状图已保存: {tree_path}")
print(f"👉 画像热力图已保存: {heat_path}")