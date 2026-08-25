import pandas as pd
import bambi as bmb
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import os

# ==========================================
# 🛑 核心控制开关 🛑
# ==========================================
# True  = 测试模式：只用 1000 条数据，5分钟跑完，用于看画图效果。
# False = 正式模式：用 2.7 万条全量数据，需要 1-2 小时，用于生成论文最终结果。
TEST_MODE = False


# ==========================================

def run_ultimate_null_model():
    print("================ 阶段 1: 数据加载 ================")
    df = pd.read_excel("ratings_for_bayesian_model.xlsx")

    # 根据模式决定数据量和采样参数
    if TEST_MODE:
        print("\n⚠️ 警告：当前开启【短时间测试模式】！")
        df = df.sample(n=1000, random_state=42)
        tune_steps = 500
        draw_steps = 500
    else:
        print("\n🚀 当前开启【全量正式模式】！请保持电脑通电。")
        tune_steps = 2000
        draw_steps = 1000

    df['rater'] = df['rater'].astype(str)
    df['image'] = df['image'].astype(str)
    categories = sorted(df['rating'].unique())
    df['rating'] = pd.Categorical(df['rating'], categories=categories, ordered=True)

    print(f"数据加载完毕。本次将使用 {len(df)} 条打分记录进行计算。\n")

    print("================ 阶段 2: 构建并拟合空模型 ================")
    model = bmb.Model("rating ~ 1 + (1|rater) + (1|image)", data=df, family="cumulative")

    results = model.fit(
        draws=draw_steps,
        tune=tune_steps,
        chains=4,
        target_accept=0.95
    )
    print("\n🎉 空模型 MCMC 采样成功完成！")

    print("\n================ 阶段 3: 计算核心学术指标 ================")
    summary = az.summary(results)

    # --- 安全计算 WAIC/LOO 模块 (容错机制) ---
    print("正在尝试计算对数似然、WAIC 和 LOO...")
    waic_str = "无法计算 (Cumulative 模型特性)"
    loo_str = "无法计算 (Cumulative 模型特性)"
    try:
        import pymc as pm
        if not hasattr(results, "log_likelihood"):
            pm.compute_log_likelihood(results, model=model.backend.model)
        waic_data = az.waic(results)
        loo_data = az.loo(results)
        waic_str = f"{waic_data.waic:.2f} (SE: {waic_data.waic_se:.2f})"
        loo_str = f"{loo_data.loo:.2f} (SE: {loo_data.loo_se:.2f})"
        print("✅ WAIC/LOO 计算成功！")
    except Exception as e:
        print(f"\n⚠️ 提示：受限于 cumulative 模型特性，跳过 WAIC/LOO 计算，直接采用 ICC。")

    # 计算 ICC (组内相关系数)
    sigma_image = summary.loc['1|image_sigma', 'mean']
    sigma_rater = summary.loc['1|rater_sigma', 'mean']
    var_image = sigma_image ** 2
    var_rater = sigma_rater ** 2
    icc_image = var_image / (var_image + var_rater)

    metrics_text = (
        "========== Null Model Baseline Metrics (空模型基准指标) ==========\n\n"
        "1. Variance Components (方差成分):\n"
        f"   - Image Variance (图片审美客观方差, σ^2): {var_image:.4f}\n"
        f"   - Rater Variance (评分人主观偏好方差, σ^2): {var_rater:.4f}\n\n"
        "2. Explanatory Baseline (解释力基准):\n"
        f"   - ICC_image (客观美感占比): {icc_image:.2%} \n\n"
        "3. Model Comparison Metrics (模型信息准则):\n"
        f"   - WAIC: {waic_str}\n"
        f"   - LOO:  {loo_str}\n"
    )
    print("\n" + metrics_text)

    print("\n================ 阶段 4: 全方位保存结果 ================")
    save_dir = "BYS_kong_2_result"
    os.makedirs(save_dir, exist_ok=True)

    summary.to_csv(f"{save_dir}/Null_Model_Summary.csv")
    with open(f"{save_dir}/Null_Model_Metrics_Report.txt", "w", encoding="utf-8") as f:
        f.write(metrics_text)

    # 强制在测试模式下也保存模型文件，验证通道是否畅通！
    print("正在保存模型时间胶囊（.nc 轨迹）...")
    results.to_netcdf(f"{save_dir}/null_model_trace.nc")
    print("✅ 模型时间胶囊 (.nc) 已永久保存！")

    print("\n================ 阶段 5: 生成清爽版高级图表 ================")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    sns.set_context("paper", font_scale=1.2)
    sns.set_style("ticks", {"font.family": "serif", "font.serif": ["Times New Roman"]})

    # (a) 全局方差迹线图
    print("正在生成图 1: 全局方差迹线图...")
    axes_trace = az.plot_trace(results, var_names=["1|rater_sigma", "1|image_sigma"])
    if isinstance(axes_trace, np.ndarray):
        for row in axes_trace:
            for ax in row:
                ax.set_yticklabels([])
    else:
        for ax in axes_trace:
            ax.set_yticklabels([])
    plt.suptitle("Trace Plots for Global Variances (Null Model)", fontsize=16, weight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/Null_Fig1_TracePlot.png", dpi=300, bbox_inches='tight')
    plt.close()

    # (b) 图片真实颜值森林图 (彻底去文字版)
    print("正在生成图 2: 图片真实颜值森林图...")
    axes_img = az.plot_forest(results, var_names=["1|image"], combined=True, hdi_prob=0.95, figsize=(8, 12))
    if isinstance(axes_img, np.ndarray):
        for ax in axes_img.flatten():
            ax.set_yticklabels([])  # 清除密集 ID
            ax.set_ylabel("")  # 彻底清除 "Images (Sorted by ID)" 侧边文字
    else:
        axes_img.set_yticklabels([])
        axes_img.set_ylabel("")
    plt.title("Intrinsic Aesthetic Scores Distribution", fontsize=16, weight='bold')
    plt.axvline(0, color='red', linestyle='--')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/Null_Fig2_ImageScores_Clean.png", dpi=300, bbox_inches='tight')
    plt.close()

    # (c) 评分人偏好森林图 (彻底去文字版)
    print("正在生成图 3: 评分人主观偏好森林图...")
    axes_rater = az.plot_forest(results, var_names=["1|rater"], combined=True, hdi_prob=0.95, figsize=(8, 15))
    if isinstance(axes_rater, np.ndarray):
        for ax in axes_rater.flatten():
            ax.set_yticklabels([])  # 清除密集 ID
            ax.set_ylabel("")  # 彻底清除 "Raters (Sorted by ID)" 侧边文字
    else:
        axes_rater.set_yticklabels([])
        axes_rater.set_ylabel("")
    plt.title("Rater Strictness/Leniency Distribution", fontsize=16, weight='bold')
    plt.axvline(0, color='red', linestyle='--')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/Null_Fig3_RaterScores_Clean.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✅ 所有测试任务完成！请检查 {os.path.abspath(save_dir)} 文件夹。")


if __name__ == '__main__':
    from multiprocessing import freeze_support

    freeze_support()
    run_ultimate_null_model()