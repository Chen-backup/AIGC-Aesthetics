import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde
import os

print("\n" + "=" * 60)
print("Start rendering heterogeneity evidence figures")
print("=" * 60)

# ==========================================
# 1. 基础配置
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

SAVE_DIR = "BYS_Heterogeneity_Evidence"
os.makedirs(SAVE_DIR, exist_ok=True)

CSV_PATH = os.path.join("BYS_Clustering_Results_Advanced", "Rater_14D_Preferences.csv")

try:
    df_preferences = pd.read_csv(CSV_PATH, index_col=0)
    # 保持 df_raw 的绝对纯净，只包含 14 个特征列
    features_only = [col for col in df_preferences.columns if col not in ['Cluster', 'tSNE_1', 'tSNE_2', 'Final_Cluster']]
    df_raw = df_preferences[features_only]
    print(f"Loaded preference matrix: {len(df_raw)} raters x {df_raw.shape[1]} features")
except FileNotFoundError:
    print(f"File not found: {CSV_PATH}")
    exit()

# ==========================================
# 图表 1：平行坐标图 (Parallel Coordinates Plot)
# ==========================================
print("\nRendering Plot1: parallel coordinates...")
FACE_MAIN_COLOR = "#88A0CB"
FACE_ENVELOPE_COLOR = "#88A0CB"
mpl.rcParams.update(
    {
        "font.family": "Times New Roman",
        "axes.linewidth": 0.9,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

fig, ax = plt.subplots(figsize=(10.8, 5.2), dpi=450)
x = np.arange(df_raw.shape[1])
values = df_raw.to_numpy(dtype=float)
q05 = df_raw.quantile(0.05).to_numpy(dtype=float)
q50 = df_raw.quantile(0.50).to_numpy(dtype=float)
q95 = df_raw.quantile(0.95).to_numpy(dtype=float)

for row in values:
    ax.plot(x, row, color=FACE_MAIN_COLOR, alpha=0.055, lw=0.78, zorder=1)

ax.fill_between(x, q05, q95, color=FACE_ENVELOPE_COLOR, alpha=0.22, linewidth=0, zorder=2)
ax.plot(x, q50, color="#202020", lw=1.75, marker="o", markersize=3.4, zorder=4, label="Median")
ax.axhline(0, color="#A23B3B", linestyle="--", linewidth=1.85, alpha=0.98, label="Neutral Baseline (0)", zorder=3)

flat_values = values.reshape(-1)
y_min = float(np.quantile(flat_values, 0.003))
y_max = float(np.quantile(flat_values, 0.997))
pad = (y_max - y_min) * 0.08
ax.set_ylim(y_min - pad, y_max + pad)

ax.set_xticks(x)
ax.set_xticklabels(df_raw.columns, rotation=45, ha="right", fontsize=10, fontweight="bold")
ax.set_xlim(0, len(df_raw.columns) - 1)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_title("Parallel Coordinates of 1300 Raters across 14 Aesthetic Features", fontsize=16, fontweight="bold", pad=12)
ax.set_ylabel("Preference Slope (BLUPs)", fontsize=12, fontweight="bold")
ax.tick_params(axis="y", labelsize=10)
for tick in ax.get_yticklabels():
    tick.set_fontweight("bold")
for xi in x:
    ax.axvline(xi, color="#CBD5DF", linestyle=":", lw=0.8, alpha=0.85, zorder=0)
ax.grid(axis="y", color="#E5E9EF", lw=0.7, alpha=0.95)
ax.set_axisbelow(True)
legend_handles = [
    mpl.lines.Line2D([0], [0], color=FACE_MAIN_COLOR, lw=1.1, alpha=0.42, label="Individual raters"),
    mpl.patches.Patch(facecolor=FACE_ENVELOPE_COLOR, edgecolor="none", alpha=0.22, label="5-95% envelope"),
    mpl.lines.Line2D([0], [0], color="#202020", lw=1.75, marker="o", markersize=3.4, label="Median"),
    mpl.lines.Line2D([0], [0], color="#A23B3B", lw=1.85, linestyle="--", label="Neutral Baseline (0)"),
]
ax.legend(handles=legend_handles, frameon=False, loc="upper right", fontsize=8.4, ncol=2, columnspacing=0.8, handlelength=1.8)

fig.subplots_adjust(left=0.075, right=0.985, bottom=0.27, top=0.89)
fig.savefig(os.path.join(SAVE_DIR, "Plot1_Parallel_Coordinates.png"), dpi=450, bbox_inches="tight", pad_inches=0.04)
plt.close(fig)

# ==========================================
# 图表 2：连续梯度热力图 (PCA 满血归一化排序版)
# ==========================================
print("Rendering Plot2: PCA-sorted heatmap...")

# 1. 对特征进行 Z-score 归一化 (解决颜色过浅的问题)
scaler = StandardScaler()
scaled_values = scaler.fit_transform(df_raw)
df_scaled = pd.DataFrame(scaled_values, index=df_raw.index, columns=df_raw.columns)

# 2. 提取第一主成分 (PC1)
pca_1d = PCA(n_components=1)
pc1_scores = pca_1d.fit_transform(df_scaled)

# 3. 对人（行）进行排序：按 PC1 得分从高到低
df_scaled['PC1_Score'] = pc1_scores
df_sorted = df_scaled.sort_values(by='PC1_Score', ascending=False).drop(columns=['PC1_Score'])

# 4. 对特征（列）进行排序：按 PC1 的权重贡献 (Loadings)
loadings = pd.Series(pca_1d.components_[0], index=df_raw.columns)
sorted_features = loadings.sort_values(ascending=False).index
df_sorted = df_sorted[sorted_features]

plt.figure(figsize=(12, 10))
# vmin=-3, vmax=3 保证极端偏好也能展现出最浓郁的深红/深蓝色
ax = sns.heatmap(df_sorted, cmap="RdBu_r", center=0, vmin=-3, vmax=3,
                 yticklabels=False, xticklabels=True,
                 cbar_kws={'label': 'Z-scored Preference Strength', 'shrink': 0.7, 'pad': 0.02})

ax.set_xlabel('14 Aesthetic Features (Sorted by PC1 Loading)', fontsize=14, weight='bold')
ax.set_ylabel(f'{len(df_raw)} Raters (Sorted Top-to-Bottom by PC1 Score)', fontsize=14, weight='bold')
plt.title(f'PCA-Sorted Continuous Gradient Heatmap of Aesthetic Preferences', fontsize=18, weight='bold', pad=20)
plt.xticks(rotation=45, ha='right', fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "Plot2_PCA_Sorted_Heatmap.png"), dpi=400)

# ==========================================
# 图表 3：PCA/UMAP 密度星系图 (Density Galaxy Scatter)
# ==========================================
print("Rendering Plot3: PCA density plot...")
pca_2d = PCA(n_components=2)
xy_pca = pca_2d.fit_transform(df_raw) # 直接使用纯净的 df_raw

x, y = xy_pca[:, 0], xy_pca[:, 1]
xy = np.vstack([x, y])
z = gaussian_kde(xy)(xy)

idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]

plt.figure(figsize=(9, 7))
scatter = plt.scatter(x, y, c=z, s=50, cmap='magma', edgecolor='none', alpha=0.8)
plt.colorbar(scatter, label='Local Population Density')
plt.title('Aesthetic Preference "Galaxy" (PCA Density Plot without Hard Clusters)', fontsize=16, weight='bold')
plt.xlabel(f'Principal Component 1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
plt.ylabel(f'Principal Component 2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "Plot3_Galaxy_Density.png"), dpi=300)

# ==========================================
# 图表 4：统计学铁证 - 误差棒森林图 (Forest Plot of Heterogeneity)
# ==========================================
print("Rendering Plot4: heterogeneity forest plot...")

means = df_raw.mean()
stds = df_raw.std()
stats_df = pd.DataFrame({'Mean': means, 'SD': stds}).sort_values(by='SD', ascending=True)

plt.figure(figsize=(10, 8))

plt.errorbar(stats_df['Mean'], range(len(stats_df)),
             xerr=1.96 * stats_df['SD'], fmt='o',
             color='#2c3e50', ecolor='#e74c3c', elinewidth=2.5, capsize=5, markersize=8)

plt.axvline(0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

plt.yticks(range(len(stats_df)), stats_df.index, fontsize=12, weight='bold')
plt.xlabel('Individual Preference Slope (Mean ± 1.96 SD)', fontsize=14, weight='bold')
plt.title('Forest Plot of Aesthetic Heterogeneity\n(Wide error bars crossing zero indicate high population disagreement)',
          fontsize=16, weight='bold', pad=15)

for i, (mean, sd) in enumerate(zip(stats_df['Mean'], stats_df['SD'])):
    plt.text(stats_df['Mean'].max() + stats_df['SD'].max()*2.2, i,
             f"SD: {sd:.3f}", va='center', fontsize=11, color='#e74c3c', weight='bold')

plt.xlim(stats_df['Mean'].min() - stats_df['SD'].max()*2.5,
         stats_df['Mean'].max() + stats_df['SD'].max()*3.5)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "Plot4_Heterogeneity_Forest.png"), dpi=300)

print("\nAll heterogeneity evidence figures have been generated.")
print(f"Output directory: {SAVE_DIR}")
