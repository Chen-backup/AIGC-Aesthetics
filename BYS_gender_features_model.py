import os
import numpy as np
import pandas as pd
import bambi as bmb
import arviz as az
import matplotlib.pyplot as plt

# ==========================================
# 1. 核心控制开关
# ==========================================
# 🛑 True  = 极速测试模式 (跑线性模型，用于快速看趋势)
# 🛑 False = 完整实验模式 (跑非线性样条模型，用于发论文，建议夜间运行)
IS_QUICK_TEST = False

# 数据路径配置
EXCEL_RATINGS = "ratings_for_bayesian_model.xlsx"
EXCEL_MAPPING = "renumber&gender.xlsx"
CSV_GEOM = "interpretable_face_features.csv"

# 结果统一输出文件夹
OUTPUT_DIR = "Gender_Gaze_Results"

# 全部 14 个可解释几何/质量特征
GEOM_FEATURES = ['face_hw_ratio', 'eye_face_w_ratio', 'mouth_face_w_ratio', 'three_courts_balance',
                 'upper_lower_ratio', 'eye_y_ratio', 'total_symmetry', 'le_nose_re_angle',
                 'mouth_nose_ratio', 'face_brightness', 'face_contrast', 'face_clarity',
                 'saturation', 'edge_density']

# 🌟 本次研究的 4 大核心焦点特征
FOCUS_FEATURES = [
    'upper_lower_ratio',
    'face_hw_ratio',
    'mouth_nose_ratio',
    'eye_face_w_ratio'
]

# 自动计算其余特征作为控制变量 (不参与交互)
OTHER_FEATURES = [f for f in GEOM_FEATURES if f not in FOCUS_FEATURES]

# ==========================================
# 2. 核心程序入口 (防 Windows 多进程崩溃)
# ==========================================
if __name__ == '__main__':
    # 自动创建输出文件夹
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    mode_name = "【极速测试模式】" if IS_QUICK_TEST else "【完整实验模式】"
    print("\n" + "=" * 60)
    print(f"🚀 启动 {mode_name} 男女审美异质性分析流...")
    print(f"📂 目标特征: {FOCUS_FEATURES}")
    print(f"📁 结果存储路径: {OUTPUT_DIR}/")
    print("=" * 60)

    # --- 2.1 数据加载与合并 ---
    df_ratings = pd.read_excel(EXCEL_RATINGS)
    df_mapping = pd.read_excel(EXCEL_MAPPING)
    df_geom = pd.read_csv(CSV_GEOM)

    geom_key = 'face_id' if 'face_id' in df_geom.columns else 'image_name'
    mapping_key = 'face_id' if 'face_id' in df_mapping.columns else df_mapping.columns[0]

    df_combined = pd.merge(df_geom, df_mapping[[mapping_key, 'Number']], left_on=geom_key, right_on=mapping_key,
                           how='inner')
    df = pd.merge(df_ratings, df_combined, left_on='image', right_on='Number', how='inner')

    # 清洗缺失值
    df = df.dropna(subset=GEOM_FEATURES + ['rating', 'rater', 'image'])

    # --- 2.2 划分打分人性别 ---
    # 1-627 女性 (Female), 628及以上 男性 (Male)
    df['rater_num'] = pd.to_numeric(df['rater'], errors='coerce')
    df['rater_gender'] = np.where(df['rater_num'] <= 627, 'Female', 'Male')

    # 处理评分为有序分类变量
    categories = sorted(df['rating'].unique())
    df['rating'] = pd.Categorical(df['rating'], categories=categories, ordered=True)
    rating_cats_numeric = np.array(categories).astype(float)

    # 记录每个特征的原始物理边界用于绘图
    real_bounds = {feat: (df[feat].min(), df[feat].max()) for feat in FOCUS_FEATURES}

    # 特征标准化 (Z-score)，加速贝叶斯收敛
    training_stats = {}
    for feat in GEOM_FEATURES:
        m, s = df[feat].mean(), df[feat].std()
        training_stats[feat] = {'mean': m, 'std': s}
        df[feat] = (df[feat] - m) / s

    # 确保分类变量类型正确
    df['rater'] = df['rater'].astype(str)
    df['image'] = df['image'].astype(str)
    df['rater_gender'] = df['rater_gender'].astype(str)

    print(
        f"✅ 数据处理完成。女性评分: {len(df[df['rater_gender'] == 'Female'])} 条，男性评分: {len(df[df['rater_gender'] == 'Male'])} 条。")

    # ==========================================
    # 3. 动态构建贝叶斯公式与拟合
    # ==========================================
    if IS_QUICK_TEST:
        # 极速版：线性交互，步数极少，主要为了看趋势
        interaction_terms = " + ".join([f"{f} * rater_gender" for f in FOCUS_FEATURES])
        draws_n, tune_n, chains_n = 100, 50, 4
        file_prefix = "LITE"
    else:
        # 完整版：bs() 非线性样条交互，步数多，收敛平滑
        interaction_terms = " + ".join([f"bs({f}, df=4) * rater_gender" for f in FOCUS_FEATURES])
        draws_n, tune_n, chains_n = 1500, 1000, 4
        file_prefix = "FULL"

    formula = f"rating ~ 1 + {' + '.join(OTHER_FEATURES)} + {interaction_terms} + (1|rater) + (1|image)"

    print(f"\n🔥 拟合公式: \n{formula}")
    model = bmb.Model(formula, data=df, family="cumulative")
    results = model.fit(draws=draws_n, tune=tune_n, chains=chains_n, target_accept=0.95, init="adapt_diag")

    # 保存模型
    save_path = os.path.join(OUTPUT_DIR, f"Gender_Gaze_{file_prefix}_Trace.nc")
    az.to_netcdf(results, save_path)
    print(f"💾 模型已保存至: {save_path}")

    # 打印统计摘要
    summary_df = az.summary(results)
    interaction_rows = [idx for idx in summary_df.index if ':' in idx]
    print("\n🏆 核心交互项统计 (HDI区间不含0代表群体差异显著):")
    print(summary_df.loc[interaction_rows, ['mean', 'hdi_3%', 'hdi_97%']])

    # ==========================================
    # 4. 生成 4 大特征的对比曲线图
    # ==========================================
    print("\n🎨 正在绘制群体审美差异对比图...")


    def plot_gender_gaze(feat):
        # 生成预测用的虚拟数据 (100个采样点)
        x_real = np.linspace(real_bounds[feat][0], real_bounds[feat][1], 100)
        x_std = (x_real - training_stats[feat]['mean']) / training_stats[feat]['std']

        # 模板：其余特征均固定在均值(0)
        base = {f: np.zeros(100) for f in GEOM_FEATURES}
        base[feat] = x_std
        base['rater'] = ["unknown"] * 100
        base['image'] = ["unknown"] * 100

        dummy_m = pd.DataFrame(base.copy());
        dummy_m['rater_gender'] = "Male"
        dummy_f = pd.DataFrame(base.copy());
        dummy_f['rater_gender'] = "Female"

        # 预测期望评分
        pred_m = model.predict(results, data=dummy_m, kind="response_params", include_group_specific=False,
                               inplace=False)
        pred_f = model.predict(results, data=dummy_f, kind="response_params", include_group_specific=False,
                               inplace=False)

        probs_m = pred_m.posterior['p'].values if 'p' in pred_m.posterior.data_vars else pred_m.posterior[
            'rating_probs'].values
        probs_f = pred_f.posterior['p'].values if 'p' in pred_f.posterior.data_vars else pred_f.posterior[
            'rating_probs'].values

        exp_m = np.sum(probs_m * rating_cats_numeric, axis=-1)
        exp_f = np.sum(probs_f * rating_cats_numeric, axis=-1)

        # 计算均值与 95% HDI 边界
        m_mean, m_l, m_u = exp_m.mean(axis=(0, 1)), np.percentile(exp_m, 2.5, axis=(0, 1)), np.percentile(exp_m, 97.5,
                                                                                                          axis=(0, 1))
        f_mean, f_l, f_u = exp_f.mean(axis=(0, 1)), np.percentile(exp_f, 2.5, axis=(0, 1)), np.percentile(exp_f, 97.5,
                                                                                                          axis=(0, 1))

        plt.figure(figsize=(9, 6))
        plt.rcParams['font.family'] = 'serif'

        # 男性曲线 (蓝色)
        plt.plot(x_real, m_mean, color='#1f77b4', linewidth=3, label="Male Raters")
        plt.fill_between(x_real, m_l, m_u, color='#1f77b4', alpha=0.15)

        # 女性曲线 (粉色)
        plt.plot(x_real, f_mean, color='#e377c2', linewidth=3, label="Female Raters")
        plt.fill_between(x_real, f_l, f_u, color='#e377c2', alpha=0.15)

        plt.title(f"[{file_prefix}] Aesthetic Gender Gaze: {feat}", fontsize=15, weight='bold')
        plt.xlabel(f"Real Values of {feat}", fontsize=13)
        plt.ylabel("Expected Aesthetic Score (1-7)", fontsize=13)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(fontsize=11)
        plt.tight_layout()

        fig_save_path = os.path.join(OUTPUT_DIR, f"Comparison_{file_prefix}_{feat}.png")
        plt.savefig(fig_save_path, dpi=300)
        print(f"✅ 图表已保存: {fig_save_path}")


    # 循环生成 4 张图
    for focus_feat in FOCUS_FEATURES:
        plot_gender_gaze(focus_feat)

    print(f"\n🏆 全部实验流程执行完毕！请在 {OUTPUT_DIR}/ 文件夹下查看结果。")