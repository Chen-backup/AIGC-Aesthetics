import pandas as pd
import bambi as bmb
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# ==========================================
TEST_MODE = False  # 测试无误后，改为 False 开启全量正式跑！
# ==========================================

def run_stylegan_model_14():
    print("================ 阶段 1: 加载 StyleGAN 14 维流形特征并拼表 ================")
    df_ratings = pd.read_excel("ratings_for_bayesian_model.xlsx")

    # 🎯 读取刚刚降维好的 StyleGAN 14 维特征表
    df_features = pd.read_csv("PCA_14_stylegan_w.csv")
    df_mapping = pd.read_excel("renumber&gender.xlsx")

    # 拼表逻辑保持不变
    df_feat_mapped = pd.merge(df_features, df_mapping[['face_id', 'Number']], left_on='image_name', right_on='face_id',
                              how='inner')
    df = pd.merge(df_ratings, df_feat_mapped, left_on='image', right_on='Number', how='inner')

    df = df.dropna()
    print(f"数据拼装完成！合并后总计 {len(df)} 条有效打分记录。")

    # ======== 👇 测试模式控制逻辑 👇 ========
    if TEST_MODE:
        print("\n⚠️ 当前为【小批量极速测试模式】(随机抽取 1000 条)")
        df = df.sample(n=1000, random_state=42)
        tune_steps, draw_steps = 500, 500
    else:
        print("\n🚀 当前为【全量正式模式】！即将进行同等复杂度的极峰对决！")
        tune_steps, draw_steps = 2000, 1000
    # ======== 👆 测试模式控制逻辑 👆 ========

    print("\n================ 阶段 2: 生成特征标准化 ================")
    pc_features = [f'PC{i + 1}' for i in range(14)]

    df['rater'] = df['rater'].astype(str)
    df['image'] = df['image'].astype(str)
    categories = sorted(df['rating'].unique())
    df['rating'] = pd.Categorical(df['rating'], categories=categories, ordered=True)

    # Z-score 标准化
    for feat in pc_features:
        df[feat] = (df[feat] - df[feat].mean()) / df[feat].std()

    print("\n================ 阶段 3: 启动 StyleGAN 14 维贝叶斯 MCMC ================")
    formula = f"rating ~ 1 + {' + '.join(pc_features)} + (1|rater) + (1|image)"
    print(f"即将运行公式:\n{formula}\n")

    model = bmb.Model(formula, data=df, family="cumulative")
    results = model.fit(draws=draw_steps, tune=tune_steps, chains=4, target_accept=0.95)
    print("\n🎉 StyleGAN 14维模型 MCMC 采样成功完成！")

    print("\n================ 阶段 4: 计算生成流形先验核心指标 ================")
    summary = az.summary(results)

    sigma_image = summary.loc['1|image_sigma', 'mean']
    sigma_rater = summary.loc['1|rater_sigma', 'mean']
    var_image = sigma_image ** 2
    var_rater = sigma_rater ** 2

    # 空模型图片方差基准
    NULL_VAR_IMAGE = 6.2101
    marginal_r2 = (NULL_VAR_IMAGE - var_image) / NULL_VAR_IMAGE

    metrics_text = (
        "========== StyleGAN-W+ Model Metrics (14维生成流形特征指标) ==========\n\n"
        "1. Variance Components (加入StyleGAN流形特征后的剩余方差):\n"
        f"   - Residual Image Variance (剩余图片方差, σ^2): {var_image:.4f} \n"
        f"   - Rater Variance (评分人主观偏好方差, σ^2): {var_rater:.4f}\n\n"
        "2. The Ultimate Generative Prior (生成先验解释力):\n"
        f"   - 空模型绝对客观方差: {NULL_VAR_IMAGE}\n"
        f"   - StyleGAN 14维特征解释力 (Marginal R²): {marginal_r2:.2%} \n"
    )
    print("\n" + metrics_text)

    print("\n================ 阶段 5: 全方位保存数据与诊断图表 ================")
    save_dir = "BYS_StyleGAN_model_14D_result"
    os.makedirs(save_dir, exist_ok=True)

    summary.to_csv(f"{save_dir}/StyleGAN_14D_Model_Summary.csv")
    with open(f"{save_dir}/StyleGAN_14D_Model_Metrics_Report.txt", "w", encoding="utf-8") as f:
        f.write(metrics_text)

    results.to_netcdf(f"{save_dir}/StyleGAN_14d_model_trace.nc")

    # --- 图表 1：14维主成分影响力的森林图 ---
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    sns.set_context("paper", font_scale=1.2)

    feat_summary = az.summary(results, var_names=pc_features)
    feat_summary_sorted = feat_summary.sort_values(by='mean', ascending=True)
    sorted_features = feat_summary_sorted.index.tolist()

    fig, ax = plt.subplots(figsize=(10, 8))
    means = feat_summary_sorted['mean']
    hdi_lower = feat_summary_sorted['hdi_3%']
    hdi_upper = feat_summary_sorted['hdi_97%']
    y_pos = np.arange(len(sorted_features))

    # 使用代表潜空间的科技紫配色
    ax.errorbar(means, y_pos, xerr=[means - hdi_lower, hdi_upper - means],
                fmt='o', color='#9b59b6', ecolor='#9b59b6', elinewidth=2, capsize=4)
    ax.axvline(x=0, color='#d62728', linestyle='--', linewidth=1.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_features)
    ax.set_xlabel('Standardized Effect Size (Posterior Mean with 95% HDI)', weight='bold')
    ax.set_title('Impact of 14 StyleGAN-W+ Deep PCs on Aesthetics', weight='bold', pad=15)
    sns.despine()
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/StyleGAN_14D_Fig1_Forest_Plot.png', dpi=300, bbox_inches='tight')
    plt.close()

    # --- 图表 2：MCMC 收敛性诊断迹线图 ---
    print("\n正在生成 MCMC 收敛性诊断迹线图 (请稍候)...")
    trace_dir = os.path.join(save_dir, "TracePlots_Diagnostics")
    os.makedirs(trace_dir, exist_ok=True)

    vars_to_plot = pc_features + ["1|rater_sigma", "1|image_sigma"]

    for var in vars_to_plot:
        axes_trace = az.plot_trace(results, var_names=[var])
        if isinstance(axes_trace, np.ndarray):
            for row in axes_trace:
                for ax in row:
                    ax.set_yticklabels([])
        else:
            for ax in axes_trace:
                ax.set_yticklabels([])

        plt.suptitle(f"Trace Plot: {var}", fontsize=16, weight='bold', y=1.05)
        plt.tight_layout()

        safe_var_name = var.replace("|", "_")
        plt.savefig(f'{trace_dir}/Trace_{safe_var_name}.png', dpi=300, bbox_inches='tight')
        plt.close()

    print(f"✅ {len(vars_to_plot)} 张独立迹线图已成功生成并存入 {trace_dir} 文件夹！")
    print(f"\n🚀🚀🚀 全部完成！最终战报已存入：{os.path.abspath(save_dir)}")


if __name__ == '__main__':
    from multiprocessing import freeze_support

    freeze_support()
    run_stylegan_model_14()