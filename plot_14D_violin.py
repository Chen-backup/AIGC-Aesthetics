import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("\n" + "★" * 60)
print("🎻 启动【14维可解释特征：多峰形态小提琴图扫描】")
print("★" * 60)

# ==========================================
# 1. 全局配置与中文字体修复
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

SAVE_DIR = "BYS_Clustering_Results_Advanced"
os.makedirs(SAVE_DIR, exist_ok=True)

# 🎯 完整的 14 个可解释特征
TARGET_FEATURES = [
    'face_hw_ratio', 'eye_face_w_ratio', 'mouth_face_w_ratio', 'three_courts_balance',
    'upper_lower_ratio', 'eye_y_ratio', 'total_symmetry', 'le_nose_re_angle',
    'mouth_nose_ratio', 'face_brightness', 'face_contrast', 'face_clarity',
    'saturation', 'edge_density'
]

# ==========================================
# 2. 加载原始几何特征数据
# ==========================================
print("⏳ 正在加载原始面部特征数据...")
try:
    df_geom = pd.read_csv("interpretable_face_features.csv")
except FileNotFoundError:
    print("❌ 找不到文件 interpretable_face_features.csv，请检查路径！")
    exit()

# 提取这 14 个特征列并剔除缺失值
df_features = df_geom[TARGET_FEATURES].dropna()
print(f"✅ 数据加载成功！共提取 {len(df_features)} 张有效面孔数据。")

# ==========================================
# 3. 绘制 4x4 小提琴图矩阵
# ==========================================
print("🎨 正在渲染 14 维度小提琴分布图...")

# 创建 4 行 4 列的画布 (最后两个图表位留空)
fig, axes = plt.subplots(4, 4, figsize=(18, 14))
axes = axes.flatten()

# 自定义一种高级的学术调色盘
palette = sns.color_palette("husl", 14)

for i, feat in enumerate(TARGET_FEATURES):
    ax = axes[i]

    # 绘制小提琴图
    # inner="quartile" 会在小提琴内部画出 25%, 50%, 75% 的分位数虚线
    # bw_adjust=0.8 让核密度估计(KDE)更敏感，更容易暴露出多峰的“小凸起”
    sns.violinplot(y=df_features[feat], ax=ax, color=palette[i], inner="quartile", bw_adjust=0.8)

    ax.set_title(feat, fontsize=12, weight='bold')
    ax.set_ylabel('')  # 隐藏 y 轴标签，使画面更干净
    ax.tick_params(axis='y', labelsize=10)

# 清理最后 2 个空白的子图 (14个特征占不满 16 个格子)
for j in range(14, 16):
    fig.delaxes(axes[j])

# 添加超级大标题
plt.suptitle('Violin Plots of 14 Interpretable Facial Features (Checking for Multimodality)',
             fontsize=20, weight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为超级标题留出空间

# 保存高清图像
save_path = os.path.join(SAVE_DIR, "Feature_Violin_Plots_14D.png")
plt.savefig(save_path, dpi=300)
print(f"\n✅ 小提琴图矩阵渲染完成！已保存至: {save_path}")