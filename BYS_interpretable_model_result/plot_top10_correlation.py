import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

print("\n" + "="*50)
print("🔍 启动【Top 10 核心特征相关性耦合分析 - 全矩阵版】...")
print("="*50)

# ==========================================
# 1. 配置文件路径与特征列表
# ==========================================
CSV_FILE = "interpretable_face_features.csv"
OUTPUT_DIR = "BYS_interpretable_model_result"

# 严格按照你跑出的 Top 10 特征顺序
TARGET_FEATURES = [
    'le_nose_re_angle',       # 0: 眼鼻三角区
    'upper_lower_ratio',      # 1: 下巴比例
    'three_courts_balance',   # 2: 三庭平衡度
    'mouth_nose_ratio',       # 3: 嘴鼻宽比
    'mouth_face_w_ratio',     # 4: 嘴脸宽比
    'total_symmetry',         # 5: 绝对对称性
    'edge_density',           # 6: 边缘密度 (图像质量)
    'eye_y_ratio',            # 7: 眼高比例
    'saturation',             # 8: 色彩饱和度 (图像质量)
    'eye_face_w_ratio'        # 9: 眼脸宽比
]

# ==========================================
# 2. 数据加载与相关性计算
# ==========================================
print(f"⏳ 正在加载原始面部特征数据: {CSV_FILE}")
try:
    df = pd.read_csv(CSV_FILE)
except FileNotFoundError:
    print(f"❌ 找不到文件 {CSV_FILE}！请检查路径。")
    exit()

# 检查特征是否都在数据集中
missing_features = [f for f in TARGET_FEATURES if f not in df.columns]
if missing_features:
    print(f"❌ 数据集中缺失以下特征: {missing_features}")
    exit()

# 筛选出这 10 个特征的数据
df_target = df[TARGET_FEATURES].copy()

# 计算 Pearson 相关系数矩阵
corr_matrix = df_target.corr(method='pearson')

print("\n📊 相关系数矩阵计算完成：")
print(corr_matrix.round(3))

# ==========================================
# 3. 绘制学术级热力图 (Heatmap)
# ==========================================
print("\n🎨 正在渲染完整的 10x10 相关性热力图...")

plt.figure(figsize=(12, 10))
plt.rcParams['font.family'] = 'serif'

# 设置发散型颜色映射：蓝色代表正相关，红色代表负相关，中间为白色
cmap = sns.diverging_palette(20, 230, as_cmap=True)

# ⭐️ 核心修改：移除了 mask 参数，直接绘制完整矩阵
sns.heatmap(corr_matrix,
            cmap=cmap,
            vmax=1.0, vmin=-1.0,
            center=0,
            square=True,
            linewidths=.5,
            annot=True,          # 在格子里显示具体数值
            fmt=".2f",           # 保留两位小数
            cbar_kws={"shrink": .8, "label": "Pearson Correlation (r)"},
            annot_kws={"size": 10, "weight": "bold"})

plt.title('Complete Correlation Matrix of Top 10 Sensitive Aesthetic Features', fontsize=18, weight='bold', pad=20)

# 美化坐标轴标签
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(rotation=0, fontsize=12)

plt.tight_layout()

# 确保输出文件夹存在
os.makedirs(OUTPUT_DIR, exist_ok=True)
save_name = os.path.join(OUTPUT_DIR, "Top10_Correlation_Heatmap_Full.png")

plt.savefig(save_name, dpi=300)
print(f"\n✅ 完整相关性热力图已成功渲染并保存至: {save_name}")
