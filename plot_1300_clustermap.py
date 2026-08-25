import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list
from sklearn.preprocessing import StandardScaler
import os

print("\n" + "★" * 60)
print("🔥 启动【终极全景矩阵：特征归一化 + 聚类重排纯净热力图】")
print("★" * 60)

# ==========================================
# 1. 基础配置
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

SAVE_DIR = "BYS_Clustering_Results_Advanced"
os.makedirs(SAVE_DIR, exist_ok=True)

CSV_PATH = os.path.join(SAVE_DIR, "Rater_14D_Preferences.csv")

try:
    df_preferences = pd.read_csv(CSV_PATH, index_col=0)
    # 只保留纯粹的 14 个特征列
    features_only = [col for col in df_preferences.columns if col not in ['Cluster', 'tSNE_1', 'tSNE_2', 'Final_Cluster']]
    df_raw = df_preferences[features_only]
except FileNotFoundError:
    print(f"❌ 找不到文件 {CSV_PATH}！")
    exit()

# ==========================================
# 2. 特征级归一化 (让所有特征显色)
# ==========================================
print("\n📏 1. 正在按列进行 Z-score 标准化...")
scaler = StandardScaler()
scaled_values = scaler.fit_transform(df_raw)
df_scaled = pd.DataFrame(scaled_values, index=df_raw.index, columns=df_raw.columns)

# ==========================================
# 3. 施展魔法：聚类重排 (Seriation / Biclustering)
# ==========================================
print("🧬 2. 正在执行“聚类重排”魔法 (重新分配座位)...")

# 计算 1300 个人（行）的相似度，并提取最佳排列顺序
row_linkage = linkage(df_scaled, method='ward')
row_order = leaves_list(row_linkage)

# 计算 14 个特征（列）的相似度，并提取最佳排列顺序
col_linkage = linkage(df_scaled.T, method='ward')
col_order = leaves_list(col_linkage)

# 根据算出的最佳顺序，将整个表格彻底大洗牌！
df_sorted = df_scaled.iloc[row_order, col_order]

# ==========================================
# 4. 绘制纯净版热力图
# ==========================================
print("🎨 3. 正在渲染重排后的纯净全景热力图...")
plt.figure(figsize=(12, 16))

# vmin=-3, vmax=3 限制色彩极值，防止极端个例破坏画面对比度
ax = sns.heatmap(
    df_sorted,
    cmap="RdBu_r",
    center=0,
    vmin=-3,
    vmax=3,
    yticklabels=False,  # 隐藏 1300 个密密麻麻的名字
    xticklabels=True,
    cbar_kws={
        'label': 'Z-scored Preference Strength (Standardized)',
        'shrink': 0.6,
        'aspect': 40,
        'pad': 0.02
    }
)

# 坐标轴与标签美化
ax.set_xlabel('14 Interpretable Facial Features (Clustered & Rearranged)', fontsize=14, weight='bold', labelpad=15)
ax.set_ylabel(f'{len(df_raw)} Individual Raters (Clustered & Rearranged)', fontsize=14, weight='bold', labelpad=10)

plt.xticks(rotation=45, ha='right', fontsize=12)
plt.title(f'Rearranged Global Heatmap of Aesthetic Preferences (N={len(df_raw)})',
          fontsize=18, weight='bold', pad=20)

plt.tight_layout()

heat_path = os.path.join(SAVE_DIR, "Rearranged_Pure_1300_Heatmap.png")
plt.savefig(heat_path, dpi=400)
print(f"\n✅ 极致色彩与绝佳秩序的热力图已出炉！已保存至: {heat_path}")