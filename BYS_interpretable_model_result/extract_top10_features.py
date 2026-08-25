import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt

print("\n" + "=" * 50)
print("🔍 启动【面部特征敏感度全量扫描仪】...")
print("=" * 50)

# ==========================================
# 1. 配置文件路径 (指向正确的全模型)
# ==========================================
TRACE_FILE = "BYS_interpretable_model_result/full_model_trace.nc"

GEOM_FEATURES = [
    'face_hw_ratio', 'eye_face_w_ratio', 'mouth_face_w_ratio', 'three_courts_balance',
    'upper_lower_ratio', 'eye_y_ratio', 'total_symmetry', 'le_nose_re_angle',
    'mouth_nose_ratio', 'face_brightness', 'face_contrast', 'face_clarity',
    'saturation', 'edge_density'
]

# ==========================================
# 2. 加载模型与智能提取系数
# ==========================================
print(f"⏳ 正在唤醒模型时间胶囊: {TRACE_FILE}")
try:
    trace = az.from_netcdf(TRACE_FILE)
    summary_df = az.summary(trace)
except FileNotFoundError:
    print(f"❌ 找不到文件 {TRACE_FILE}！请检查路径是否正确。")
    exit()

print("✅ 模型加载成功！正在计算标准化特征敏感度...")

feature_stats = []
for feat in GEOM_FEATURES:
    # 筛选包含该特征名，且不包含 ':'（排除性别交互项，只看主效应）的行
    matched_rows = [idx for idx in summary_df.index if feat in idx and ':' not in idx]

    if matched_rows:
        # ⭐️ 核心逻辑：如果是 bs() 非线性样条，会有多个系数。我们取绝对值最大的那个，代表该特征的“峰值敏感度”
        max_row = max(matched_rows, key=lambda x: abs(summary_df.loc[x, 'mean']))

        mean_val = summary_df.loc[max_row, 'mean']
        hdi_lower = summary_df.loc[max_row, 'hdi_3%']
        hdi_upper = summary_df.loc[max_row, 'hdi_97%']

        feature_stats.append({
            'Feature': feat,
            'Coefficient': mean_val,
            'Sensitivity (Abs)': abs(mean_val),
            'HDI_3%': hdi_lower,
            'HDI_97%': hdi_upper,
            'Is_NonLinear': len(matched_rows) > 1
        })

if not feature_stats:
    print("❌ 在模型中未找到特征系数！")
    exit()

# 转为 DataFrame 并按绝对值降序排列
df_stats = pd.DataFrame(feature_stats)
df_sorted = df_stats.sort_values(by='Sensitivity (Abs)', ascending=False)

# 截取 Top 10
top10_df = df_sorted.head(10).reset_index(drop=True)

print("\n🏆 【Top 10 最敏感面部特征龙虎榜】")
print(top10_df[['Feature', 'Coefficient', 'Sensitivity (Abs)', 'Is_NonLinear']].to_string(index=True))

# ==========================================
# 3. 绘制学术级敏感度龙卷风图 (Tornado Plot)
# ==========================================
# ⚠️ 画布高度从 6.5 调高到 8，防止 10 个柱子显得拥挤
plt.figure(figsize=(10, 8))
plt.rcParams['font.family'] = 'serif'

# 按照敏感度从小到大排序（为了在水平柱状图上让最大的排在最上面）
plot_data = top10_df.sort_values(by='Sensitivity (Abs)', ascending=True).copy()

# 设置颜色：正向（蓝色），负向（红色）
colors = ['#1f77b4' if val > 0 else '#d62728' for val in plot_data['Coefficient']]

bars = plt.barh(plot_data['Feature'], plot_data['Coefficient'], color=colors, alpha=0.85, edgecolor='black',
                linewidth=1.2)

# 添加 0 刻度基准线
plt.axvline(0, color='black', linewidth=1.5, linestyle='-')

# 添加误差线 (95% HDI) 和数值标注
for i, bar in enumerate(bars):
    feature = plot_data['Feature'].iloc[i]
    coef = plot_data['Coefficient'].iloc[i]
    lower = plot_data['HDI_3%'].iloc[i]
    upper = plot_data['HDI_97%'].iloc[i]

    # 绘制误差线
    plt.plot([lower, upper], [i, i], color='black', linewidth=1.5, zorder=3)
    plt.scatter([lower, upper], [i, i], color='black', s=20, zorder=4)

    # 在柱子旁边标注具体数值
    align = 'left' if coef > 0 else 'right'
    offset = 0.08 if coef > 0 else -0.08
    plt.text(coef + offset, i, f'{coef:.2f}', va='center', ha=align, fontsize=11, fontweight='bold')

plt.title('Top 10 Most Sensitive Aesthetic Features\n(Max Absolute Standardized Effects)', fontsize=16,
          fontweight='bold', pad=15)
plt.xlabel('Effect Size / Peak Sensitivity (Standard Deviations)', fontsize=12)
plt.ylabel('Interpretable Features', fontsize=12)

# 添加自定义图例
import matplotlib.patches as mpatches

blue_patch = mpatches.Patch(color='#1f77b4', label='Positive Peak Impact (加分趋势)')
red_patch = mpatches.Patch(color='#d62728', label='Negative Peak Impact (扣分趋势)')
plt.legend(handles=[blue_patch, red_patch], loc='lower right', framealpha=0.95)

plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()

save_name = "BYS_interpretable_model_result/Top10_Sensitive_Features.png"
plt.savefig(save_name, dpi=300)
print(f"\n✅ Top 10 龙卷风图表已完美渲染并保存至: {save_name}")