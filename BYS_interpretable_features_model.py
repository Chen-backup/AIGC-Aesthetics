import pandas as pd
import bambi as bmb
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# ==========================================
# 🛑 核心控制开关 🛑
# ==========================================
# True  = 测试模式：抽样 1000 条，几分钟跑完验证流程与画图。
# False = 正式模式：全量数据，需几小时，用于 SCI 最终出图与出数据。
TEST_MODE = False


# ==========================================

def run_full_model():
    print("================ 阶段 1: 数据加载与智能拼装 ================")
    print("正在读取数据 (请确保三个文件都在当前目录下)...")
    df_ratings = pd.read_excel("ratings_for_bayesian_model.xlsx")
    df_features = pd.read_csv("interpretable_face_features.csv")
    df_mapping = pd.read_excel("renumber&gender.xlsx")

    print("正在进行三表合并匹配...")
    # 借用 mapping 表的桥梁作用
    df_feat_mapped = pd.merge(df_features, df_mapping[['face_id', 'Number']], left_on='image_name', right_on='face_id',
                              how='inner')
    # 将打分数据与特征数据无缝拼合
    df = pd.merge(df_ratings, df_feat_mapped, left_on='image', right_on='Number', how='inner')

    # 清洗缺失值
    initial_len = len(df)
    df = df.dropna()
    print(f"数据拼装完成！合并后总计 {len(df)} 条有效打分记录 (剔除了 {initial_len - len(df)} 条缺失值)。")

    # 根据模式决定数据量和采样参数
    if TEST_MODE:
        print("\n⚠️ 警告：当前开启【短时间测试模式】！")
        df = df.sample(n=1000, random_state=42)
        tune_steps = 500
        draw_steps = 500
    else:
        print("\n🚀 当前开启【全量正式模式】！预计耗时较长，请保持电脑通电。")
        tune_steps = 2000
        draw_steps = 1000

    print("\n================ 阶段 2: 核心特征清洗与标准化 ================")
    selected_features = [
        'face_hw_ratio', 'eye_face_w_ratio', 'mouth_face_w_ratio',
        'three_courts_balance', 'upper_lower_ratio', 'eye_y_ratio',
        'total_symmetry', 'le_nose_re_angle', 'mouth_nose_ratio',
        'face_brightness', 'face_contrast', 'face_clarity',
        'saturation', 'edge_density'
    ]

    # 数据类型处理
    df['rater'] = df['rater'].astype(str)
    df['image'] = df['image'].astype(str)
    categories = sorted(df['rating'].unique())
    df['rating'] = pd.Categorical(df['rating'], categories=categories, ordered=True)

    print("正在对 14 个特征进行标准化处理 (Z-score)...")
    for feat in selected_features:
        df[feat] = (df[feat] - df[feat].mean()) / df[feat].std()

    print("\n================ 阶段 3: 启动贝叶斯 MCMC 采样 ================")
    formula = f"rating ~ 1 + {' + '.join(selected_features)} + (1|rater) + (1|image)"
    print(f"即将运行的模型公式:\n{formula}\n")

    model = bmb.Model(formula, data=df, family="cumulative")
    results = model.fit(
        draws=draw_steps,
        tune=tune_steps,
        chains=4,
        target_accept=0.95
    )
    print("\n🎉 全模型 MCMC 采样成功完成！")

    print("\n================ 阶段 4: 计算核心学术指标 ================")
    summary = az.summary(results)

    # --- 安全计算 WAIC/LOO ---
    print("正在尝试计算 WAIC 和 LOO...")
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
        print(f"⚠️ 受限于 cumulative 模型特性，跳过 WAIC/LOO 计算。请关注方差下降。")

    # 计算方差成分 (这里的 image_variance 已经是【加入特征后的残余方差】了！)
    sigma_image = summary.loc['1|image_sigma', 'mean']
    sigma_rater = summary.loc['1|rater_sigma', 'mean']
    var_image = sigma_image ** 2
    var_rater = sigma_rater ** 2

    metrics_text = (
        "========== Full Model Metrics (14特征全模型指标) ==========\n\n"
        "1. Variance Components (加入特征后的剩余方差):\n"
        f"   - Residual Image Variance (剩余图片方差, σ^2): {var_image:.4f} \n"
        "     [💡重要：请用空模型的 4.5882 减去这个值，再除以 4.5882，来计算 Marginal R^2 (特征解释力)！]\n"
        f"   - Rater Variance (评分人主观偏好方差, σ^2): {var_rater:.4f}\n\n"
        "2. Model Comparison Metrics (模型信息准则):\n"
        f"   - WAIC: {waic_str}\n"
        f"   - LOO:  {loo_str}\n"
    )
    print("\n" + metrics_text)

    print("\n================ 阶段 5: 全方位保存数据 ================")
    save_dir = "BYS_interpretable_model_result"
    os.makedirs(save_dir, exist_ok=True)

    summary.to_csv(f"{save_dir}/Full_Model_Summary.csv")
    with open(f"{save_dir}/Full_Model_Metrics_Report.txt", "w", encoding="utf-8") as f:
        f.write(metrics_text)

    print("正在保存核心模型时间胶囊（.nc 轨迹）...")
    results.to_netcdf(f"{save_dir}/full_model_trace.nc")
    print("✅ 模型时间胶囊 (.nc) 已永久保存！")

    print("\n================ 阶段 6: SCI 学术级图表渲染 ================")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    sns.set_context("paper", font_scale=1.2)
    sns.set_style("ticks", {"font.family": "serif", "font.serif": ["Times New Roman"]})

    # 提取特征的统计量并排序
    feat_summary = az.summary(results, var_names=selected_features)
    feat_summary_sorted = feat_summary.sort_values(by='mean', ascending=True)
    sorted_features = feat_summary_sorted.index.tolist()

    # --- 图表 1：精美版特征效应森林图 ---
    print("正在生成图 1：特征效应森林图...")
    fig, ax = plt.subplots(figsize=(10, 8))
    means = feat_summary_sorted['mean']
    hdi_lower = feat_summary_sorted['hdi_3%']
    hdi_upper = feat_summary_sorted['hdi_97%']
    y_pos = np.arange(len(sorted_features))

    ax.errorbar(means, y_pos, xerr=[means - hdi_lower, hdi_upper - means],
                fmt='o', color='#1f77b4', ecolor='#1f77b4', elinewidth=2, capsize=4,
                markersize=8, markeredgecolor='white', markeredgewidth=1)
    ax.axvline(x=0, color='#d62728', linestyle='--', linewidth=1.5, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_features)
    ax.set_xlabel('Standardized Effect Size (Posterior Mean with 95% HDI)', weight='bold')
    ax.set_title('Impact of Facial Features on Aesthetic Ratings', weight='bold', pad=15)
    sns.despine()
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/Full_Fig1_Forest_Plot.png', dpi=300, bbox_inches='tight')
    plt.close()

    # --- 图表 2：后验概率密度山脊图 ---
    print("正在生成图 2：后验概率山脊图...")
    az.plot_forest(results, var_names=selected_features, kind='ridgeplot',
                   combined=True, ridgeplot_alpha=0.6, ridgeplot_overlap=1.2,
                   colors='#2ca02c', figsize=(10, 10))
    plt.title('Posterior Density Distributions of Facial Features', fontname='Times New Roman', fontsize=18,
              weight='bold')
    plt.axvline(0, color='red', linestyle='--', linewidth=1.5)
    plt.xlabel('Parameter Value', fontname='Times New Roman', fontsize=14)
    plt.ylabel('Features', fontname='Times New Roman', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/Full_Fig2_Ridge_Plot.png', dpi=300, bbox_inches='tight')
    plt.close()

    # --- 图表 3：最核心 Top 4 特征的详细后验特写 ---
    print("正在生成图 3：核心特征详细后验分布图...")
    feat_summary['abs_mean'] = feat_summary['mean'].abs()
    top_4_features = feat_summary.sort_values(by='abs_mean', ascending=False).head(4).index.tolist()

    axes = az.plot_posterior(results, var_names=top_4_features, hdi_prob=0.95, color='#8c564b', figsize=(12, 8),
                             textsize=12)
    plt.suptitle('Detailed Posterior Distributions for Top 4 Influential Features', fontname='Times New Roman',
                 fontsize=18, weight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/Full_Fig3_Top4_Posterior.png', dpi=300, bbox_inches='tight')
    plt.close()

    # --- 图表 4：全模型 16 个核心参数的独立迹线图 (MCMC 收敛性诊断) ---
    print("正在生成图 4：16 幅独立参数迹线图 (请稍候)...")

    trace_dir = os.path.join(save_dir, "TracePlots_Diagnostics")
    os.makedirs(trace_dir, exist_ok=True)

    vars_to_plot = selected_features + ["1|rater_sigma", "1|image_sigma"]

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

    print(f"✅ 16 张独立迹线图已成功生成并存入 {trace_dir} 文件夹！")
    print(f"\n🚀🚀🚀 所有任务圆满完成！你的全模型数据、胶囊与图表已存入文件夹：{os.path.abspath(save_dir)}")


if __name__ == '__main__':
    from multiprocessing import freeze_support

    freeze_support()
    run_full_model()