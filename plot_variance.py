import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# ==========================================
# 🎛️ 绘图控制台：一键切换你要画图的模型
# ==========================================
# 👉 可选: "DINOv2" 或 "InsightFace"
AI_MODEL_TYPE = "InsightFace"

SAVE_DIR = f"BYS_Fusion_28D_{AI_MODEL_TYPE}_result"
NC_FILE = f"Fusion_28D_{AI_MODEL_TYPE}_model_trace.nc"


# ==========================================

def plot_random_effects():
    print(f"================ 1. 读取 {AI_MODEL_TYPE} 专属模型时间胶囊 ================")
    nc_path = os.path.join(SAVE_DIR, NC_FILE)

    if not os.path.exists(nc_path):
        print(f"❌ 找不到文件: {nc_path}")
        return

    try:
        trace = az.from_netcdf(nc_path)
        print("✅ 成功加载模型轨迹数据！")
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        return

    print("================ 2. 提取并计算 Rater (评分人) 的个体方差/截距 ================")
    # 提取 rater 的随机效应
    rater_summary = az.summary(trace, var_names=["1|rater"])
    # 按照后验均值排序，画出来的毛毛虫图才会是从低到高极其优雅的曲线
    rater_summary_sorted = rater_summary.sort_values('mean')

    # 开始画 Rater 的毛毛虫图
    plt.figure(figsize=(10, max(6, len(rater_summary_sorted) * 0.25)))  # 动态高度
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    sns.set_context("paper", font_scale=1.2)

    means = rater_summary_sorted['mean']
    hdi_lower = rater_summary_sorted['hdi_3%']
    hdi_upper = rater_summary_sorted['hdi_97%']
    y_pos = np.arange(len(rater_summary_sorted))

    plt.errorbar(means, y_pos, xerr=[means - hdi_lower, hdi_upper - means],
                 fmt='o', color='#d62728', ecolor='#d62728', elinewidth=1.5, capsize=3, alpha=0.8)

    plt.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    plt.yticks(y_pos, rater_summary_sorted.index, fontsize=8)

    plt.title(f'Rater Random Effects (Subjective Bias)\nModel: {AI_MODEL_TYPE} Fusion', weight='bold', pad=15)
    plt.xlabel('Deviation from Average Rating (Posterior Mean with 95% HDI)', weight='bold')
    plt.ylabel('Rater ID', weight='bold')

    sns.despine()
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()

    rater_save_path = os.path.join(SAVE_DIR, f'Fig3_Caterpillar_Rater_{AI_MODEL_TYPE}.png')
    plt.savefig(rater_save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Rater 偏差图已保存至: {rater_save_path}")

    print("================ 3. 提取并计算 Image (图片) 的个体残差 ================")
    # 提取 image 的随机效应 (这可能有 297 个，图会非常长！)
    image_summary = az.summary(trace, var_names=["1|image"])
    image_summary_sorted = image_summary.sort_values('mean')

    # 因为有将近 300 张图，我们需要把画布拉得非常长，否则字会挤在一起
    plt.figure(figsize=(12, len(image_summary_sorted) * 0.15))

    means = image_summary_sorted['mean']
    hdi_lower = image_summary_sorted['hdi_3%']
    hdi_upper = image_summary_sorted['hdi_97%']
    y_pos = np.arange(len(image_summary_sorted))

    plt.errorbar(means, y_pos, xerr=[means - hdi_lower, hdi_upper - means],
                 fmt='o', color='#1f77b4', ecolor='#1f77b4', elinewidth=1, capsize=2, alpha=0.6)

    plt.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    plt.yticks(y_pos, image_summary_sorted.index, fontsize=6)  # 字体调小

    plt.title(f'Image Random Intercepts (Unexplained Aesthetic Variance)\nModel: {AI_MODEL_TYPE} Fusion', weight='bold',
              pad=15)
    plt.xlabel('Unexplained Aesthetic Score (Posterior Mean with 95% HDI)', weight='bold')
    plt.ylabel('Image ID', weight='bold')

    sns.despine()
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()

    image_save_path = os.path.join(SAVE_DIR, f'Fig4_Caterpillar_Image_{AI_MODEL_TYPE}.png')
    plt.savefig(image_save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Image 偏差图已保存至: {image_save_path}")
    print("\n🎉 全部大功告成！快去文件夹里查看这两张震撼的毛毛虫图吧！")


if __name__ == "__main__":
    plot_random_effects()