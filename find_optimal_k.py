import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import os

print("\n" + "★" * 60)
print("📐 启动【GMM 最优聚类数量 (K) 数学诊断雷达】")
print("★" * 60)

# ==========================================
# 1. 基础配置
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

SAVE_DIR = "BYS_Clustering_Results_Advanced"
os.makedirs(SAVE_DIR, exist_ok=True)

CSV_PATH = os.path.join("BYS_Clustering_Results_Advanced", "Rater_14D_Preferences.csv")

try:
    df_preferences = pd.read_csv(CSV_PATH, index_col=0)
    features_only = [col for col in df_preferences.columns if col not in ['Cluster', 'tSNE_1', 'tSNE_2']]
    X_raw = df_preferences[features_only]
    print(f"✅ 成功加载偏好数据：{len(X_raw)} 个评委，{X_raw.shape[1]} 个审美维度。")
except FileNotFoundError:
    print(f"❌ 找不到文件 {CSV_PATH}！请检查它在上一个脚本中被保存在了哪个文件夹。")
    exit()

# ==========================================
# 2. 计算 K=2 到 K=10 的各种指标
# ==========================================
print("⏳ 正在计算 K=2 到 10 的 BIC, AIC 与轮廓系数...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

k_range = range(2, 11)
bic_scores = []
aic_scores = []
silhouette_scores = []

for k in k_range:
    # 训练 GMM 模型
    gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42, n_init=5)
    labels = gmm.fit_predict(X_scaled)

    # 记录三大指标
    bic_scores.append(gmm.bic(X_scaled))
    aic_scores.append(gmm.aic(X_scaled))
    silhouette_scores.append(silhouette_score(X_scaled, labels))

# ==========================================
# ⭐️ 新增：打印具体指标数值表格供复制
# ==========================================
print("\n📊 【各 K 值聚类评估指标具体数值表】(请复制以下内容发送给我分析)")
print("-" * 65)
print(f"{'K (聚类数)':<10} | {'BIC Score (越小越好)':<20} | {'Silhouette (越大越好)'}")
print("-" * 65)
for i, k in enumerate(k_range):
    print(f"K = {k:<6} | {bic_scores[i]:<20.2f} | {silhouette_scores[i]:.4f}")
print("-" * 65)

# ==========================================
# 3. 绘制诊断双子图 (Elbow & Silhouette)
# ==========================================
print("\n🎨 正在渲染最优 K 值诊断图表...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# --- 左图：BIC & AIC ---
ax1.plot(k_range, bic_scores, marker='o', linestyle='-', linewidth=2, label='BIC (贝叶斯信息准则)', color='#e74c3c')
ax1.plot(k_range, aic_scores, marker='s', linestyle='--', linewidth=2, label='AIC (赤池信息准则)', color='#3498db')
ax1.set_title('Information Criteria (Lower is Better)', fontsize=14, weight='bold')
ax1.set_xlabel('Number of Clusters (K)', fontsize=12)
ax1.set_ylabel('Score', fontsize=12)
ax1.set_xticks(k_range)
ax1.legend()
ax1.grid(True, alpha=0.3)

# --- 右图：Silhouette Score ---
ax2.plot(k_range, silhouette_scores, marker='^', linestyle='-', linewidth=2, color='#2ecc71')
ax2.set_title('Silhouette Score (Higher is Better)', fontsize=14, weight='bold')
ax2.set_xlabel('Number of Clusters (K)', fontsize=12)
ax2.set_ylabel('Score', fontsize=12)
ax2.set_xticks(k_range)
ax2.grid(True, alpha=0.3)

plt.suptitle('GMM Optimal Cluster Selection Diagnostics', fontsize=18, weight='bold', y=1.05)
plt.tight_layout()

save_path = os.path.join(SAVE_DIR, "Optimal_K_Diagnostics.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"\n✅ 诊断图已生成！请打开查看: {save_path}")