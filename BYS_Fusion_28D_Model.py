import pandas as pd
import bambi as bmb
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from matplotlib.lines import Line2D

# ==========================================
# 🎛️ 终极融合模型控制台 (大满贯版)
# ==========================================
TEST_MODE = False  # 建议先保持 True 测试跑通，再改为 False 跑全量！
AI_MODEL_TYPE = "StyleGAN"  # 👉 在这里无缝切换！可选: "InsightFace", "DINOv2", 或 "StyleGAN"

# ⚠️ 确认包含 14 个可解释特征的文件名
GEOM_FILE = "interpretable_face_features.csv"


# ==========================================

def run_fusion_model():
    print(f"================ 阶段 1: 开启【人类手工 + {AI_MODEL_TYPE}】28维融合 ================")

    # 1. 读取基础评分数据和映射表
    df_ratings = pd.read_excel("ratings_for_bayesian_model.xlsx")
    df_mapping = pd.read_excel("renumber&gender.xlsx")

    # 2. 自动根据模型类型加载对应的 AI 特征库
    if AI_MODEL_TYPE == "DINOv2":
        pca_file = "PCA_14_dinov2.csv"
    elif AI_MODEL_TYPE == "StyleGAN":
        pca_file = "PCA_14_stylegan_w.csv"
        
    else:
        pca_file = "PCA_14_features.csv"  # 默认对应 InsightFace

    print(f"正在加载 AI 特征库: {pca_file}")
    try:
        df_ai_features = pd.read_csv(pca_file)
    except FileNotFoundError:
        print(f"❌ 找不到文件 {pca_file}！请先运行对应的特征提取和降维代码。")
        return

    # 3. 读取 14 维人类可解释特征
    print(f"正在加载人类手工特征库: {GEOM_FILE}")
    try:
        df_geom_features = pd.read_csv(GEOM_FILE)
    except FileNotFoundError:
        print(f"❌ 找不到文件 {GEOM_FILE}！请检查文件名是否正确。")
        return

    # 4. 终极大拼表
    df_ai_mapped = pd.merge(df_ai_features, df_mapping[['face_id', 'Number']], left_on='image_name', right_on='face_id',
                            how='inner')
    merge_col = 'face_id' if 'face_id' in df_geom_features.columns else 'image_name'
    df_combined_features = pd.merge(df_ai_mapped, df_geom_features, on=merge_col, how='inner')
    df = pd.merge(df_ratings, df_combined_features, left_on='image', right_on='Number', how='inner')

    # 定义我们要用的 28 个特征名
    geom_features_list = [
        'face_hw_ratio', 'eye_face_w_ratio', 'mouth_face_w_ratio',
        'three_courts_balance', 'upper_lower_ratio', 'eye_y_ratio',
        'total_symmetry', 'le_nose_re_angle', 'mouth_nose_ratio',
        'face_brightness', 'face_contrast', 'face_clarity',
        'saturation', 'edge_density'
    ]
    pc_features_list = [f'PC{i + 1}' for i in range(14)]
    all_28_features = geom_features_list + pc_features_list

    # 剔除包含这 28 个特征的缺失值
    df = df.dropna(subset=all_28_features + ['rating', 'rater', 'image'])
    print(f"拼装完成！合并后总计 {len(df)} 条有效打分记录。")

    if TEST_MODE:
        print("\n⚠️ 当前为【小批量极速测试模式】(随机抽取 1000 条)")
        df = df.sample(n=1000, random_state=42)
        tune_steps, draw_steps = 500, 500
    else:
        print(f"\n🚀 当前为【全量正式模式】！即将进行 28 维终极融合试炼！")
        tune_steps, draw_steps = 2000, 1000

    print("\n================ 阶段 2: 28个特征全面标准化 ================")
    df['rater'] = df['rater'].astype(str)
    df['image'] = df['image'].astype(str)
    categories = sorted(df['rating'].unique())
    df['rating'] = pd.Categorical(df['rating'], categories=categories, ordered=True)

    for feat in all_28_features:
        df[feat] = (df[feat] - df[feat].mean()) / df[feat].std()

    print("\n================ 阶段 3: 启动 28 维融合贝叶斯 MCMC ================")
    formula = f"rating ~ 1 + {' + '.join(all_28_features)} + (1|rater) + (1|image)"
    print(f"即将运行惊天公式 (共 28 个固定效应)...\n")

    model = bmb.Model(formula, data=df, family="cumulative")
    results = model.fit(draws=draw_steps, tune=tune_steps, chains=4, target_accept=0.95)
    print("\n🎉 28维融合模型 MCMC 采样成功完成！")

    print("\n================ 阶段 4: 计算终极解释力 ================")
    summary = az.summary(results)

    sigma_image = summary.loc['1|image_sigma', 'mean']
    sigma_rater = summary.loc['1|rater_sigma', 'mean']
    var_image = sigma_image ** 2
    var_rater = sigma_rater ** 2

    NULL_VAR_IMAGE = 6.2101
    marginal_r2 = (NULL_VAR_IMAGE - var_image) / NULL_VAR_IMAGE

    metrics_text = (
        f"========== 28D Fusion Model Metrics ({AI_MODEL_TYPE} + 可解释特征) ==========\n\n"
        "1. Variance Components:\n"
        f"   - Residual Image Variance (剩余图片方差, σ^2): {var_image:.4f} \n"
        f"   - Rater Variance (主观偏好方差, σ^2): {var_rater:.4f}\n\n"
        "2. The Ultimate Fusion Explanatory Power:\n"
        f"   - 28维融合解释力 (Marginal R^2): {marginal_r2:.2%} \n"
    )
    print("\n" + metrics_text)

    print("\n================ 阶段 5: 保存史诗级战报、模型与图表 ================")
    save_dir = f"BYS_Fusion_28D_{AI_MODEL_TYPE}_result"
    os.makedirs(save_dir, exist_ok=True)

    summary.to_csv(f"{save_dir}/Fusion_28D_Model_Summary.csv")
    with open(f"{save_dir}/Fusion_28D_Model_Metrics.txt", "w", encoding="utf-8") as f:
        f.write(metrics_text)

    nc_path = f"{save_dir}/Fusion_28D_{AI_MODEL_TYPE}_model_trace.nc"
    results.to_netcdf(nc_path)
    print(f"📦 完整模型时间胶囊已保存至: {nc_path}")

    # --- 图表 1：画一棵超大的森林图 (28 个特征) ---
    print("正在渲染 28 维参数森林图...")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    sns.set_context("paper", font_scale=1.0)

    feat_summary = az.summary(results, var_names=all_28_features)
    feat_summary_sorted = feat_summary.sort_values(by='mean', ascending=True)
    sorted_features = feat_summary_sorted.index.tolist()

    fig, ax = plt.subplots(figsize=(10, 12))
    means = feat_summary_sorted['mean']
    hdi_lower = feat_summary_sorted['hdi_3%']
    hdi_upper = feat_summary_sorted['hdi_97%']
    y_pos = np.arange(len(sorted_features))

    # 🎨 动态主题配色：根据所选大模型赋予不同的专属高亮颜色
    if AI_MODEL_TYPE == "DINOv2":
        ai_color = '#ff7f0e'  # DINOv2 专属橙色
    elif AI_MODEL_TYPE == "StyleGAN":
        ai_color = '#9b59b6'  # StyleGAN 专属科技紫
    else:
        ai_color = '#1f77b4'  # InsightFace 专属深蓝

    colors = [ai_color if feat.startswith('PC') else '#2ca02c' for feat in sorted_features]

    for i in range(len(sorted_features)):
        ax.errorbar(means.iloc[i], y_pos[i],
                    xerr=[[means.iloc[i] - hdi_lower.iloc[i]], [hdi_upper.iloc[i] - means.iloc[i]]],
                    fmt='o', color=colors[i], ecolor=colors[i], elinewidth=2, capsize=4)

    ax.axvline(x=0, color='#d62728', linestyle='--', linewidth=1.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_features)
    ax.set_xlabel('Standardized Effect Size (Posterior Mean with 95% HDI)', weight='bold')
    ax.set_title(f'Human Priors vs {AI_MODEL_TYPE} Deep Features (28D Fusion)', weight='bold', pad=15)

    custom_lines = [Line2D([0], [0], color='#2ca02c', marker='o', lw=2),
                    Line2D([0], [0], color=ai_color, marker='o', lw=2)]
    ax.legend(custom_lines, ['Human Interpretable Features', f'{AI_MODEL_TYPE} Deep PCs'], loc='lower right')

    sns.despine()
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/Fusion_28D_Fig1_Forest_Plot.png', dpi=300, bbox_inches='tight')
    plt.close()

    # --- 图表 2：生成独立的收敛诊断图 ---
    print("\n正在生成 MCMC 收敛性诊断迹线图 (共 30 张，这可能需要一两分钟)...")

    trace_dir = os.path.join(save_dir, "TracePlots_Diagnostics")
    os.makedirs(trace_dir, exist_ok=True)

    vars_to_plot = all_28_features + ["1|rater_sigma", "1|image_sigma"]

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

    print(f"✅ {len(vars_to_plot)} 张独立迹线图已成功生成！")
    print(f"\n🚀🚀🚀 融合大业全部完成！战报及模型数据已存入：{os.path.abspath(save_dir)}")


if __name__ == '__main__':
    from multiprocessing import freeze_support

    freeze_support()
    run_fusion_model()