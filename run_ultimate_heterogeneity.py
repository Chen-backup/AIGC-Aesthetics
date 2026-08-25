import pandas as pd
import bambi as bmb
import arviz as az
import numpy as np
import os
import gc  # 引入垃圾回收机制，防止连轴跑模型导致内存爆炸
import time

print("\n" + "★" * 60)
print("🚀 启动【终极严谨版】人群审美异质性全量扫描仪")
print("★" * 60)

# ==========================================
# 1. 全局核心配置
# ==========================================
# ⚠️ 强烈建议：第一次运行保持 TEST_MODE = True，跑通 1 个特征后，再改 False 挂机过夜！
TEST_MODE = False
SAVE_DIR = "BYS_Ultimate_Heterogeneity_Results"
os.makedirs(SAVE_DIR, exist_ok=True)

# 🎯 需要循环探索“审美分歧度”的核心 6 个特征
TARGET_FEATURES = [
    'le_nose_re_angle', 'upper_lower_ratio', 'mouth_face_w_ratio',
    'total_symmetry', 'edge_density', 'eye_y_ratio'
]

# 🧱 14 个可解释几何特征 (作为严格的控制主效应 X)
ALL_14_FEATURES = [
    'face_hw_ratio', 'eye_face_w_ratio', 'mouth_face_w_ratio', 'three_courts_balance',
    'upper_lower_ratio', 'eye_y_ratio', 'total_symmetry', 'le_nose_re_angle',
    'mouth_nose_ratio', 'face_brightness', 'face_contrast', 'face_clarity',
    'saturation', 'edge_density'
]

# 🕳️ 14 个 Dinov2 深度主成分 (作为潜意识/全局轮廓控制变量 PC)
PC_COLS = [f'PC{i}' for i in range(1, 15)]


def run_ultimate_models():
    print("\n================ [阶段 1] 数据重组与严密清洗 ================")
    start_time = time.time()

    # 1. 加载文件
    # (如果提示找不到文件，请确保 CSV/Excel 和此脚本在同一文件夹，或修改为绝对路径)
    try:
        df_ratings = pd.read_excel("ratings_for_bayesian_model.xlsx")
        df_mapping = pd.read_excel("renumber&gender.xlsx")
        df_geom = pd.read_csv("interpretable_face_features.csv")
        df_dinov2 = pd.read_csv("PCA_14_dinov2.csv")
    except FileNotFoundError as e:
        print(f"❌ 找不到数据文件，请检查路径: {e}")
        return

    # 2. 级联合并
    print("⏳ 正在进行高维特征的数据级联...")
    geom_key = 'face_id' if 'face_id' in df_geom.columns else 'image_name'
    dino_key = 'face_id' if 'face_id' in df_dinov2.columns else 'image_name'

    df_features = pd.merge(df_geom, df_dinov2, left_on=geom_key, right_on=dino_key, how='inner')
    df_combined = pd.merge(df_mapping[['face_id', 'Number']], df_features, left_on='face_id', right_on=geom_key,
                           how='inner')
    df = pd.merge(df_ratings, df_combined, left_on='image', right_on='Number', how='inner')

    # 3. 极严苛的缺失值剔除 (必须确保 28 个变量全都有值)
    required_cols = ALL_14_FEATURES + PC_COLS + ['rating', 'rater', 'image']
    df = df.dropna(subset=required_cols)

    # 4. 模式控制
    if TEST_MODE:
        df = df.sample(n=1000, random_state=42)
        tune, draws = 500, 500
        print(f"⚠️ 【测试模式启动】随机抽取 1000 条评价进行管线联调...")
    else:
        tune, draws = 1500, 1000
        print(f"🔥 【全量模式启动】即将对 {len(df)} 条评价展开极限运算！请确保电脑散热良好...")

    # 5. 格式化目标变量与层级变量
    df['rater'] = df['rater'].astype(str)
    df['image'] = df['image'].astype(str)

    categories = sorted(df['rating'].unique())
    df['rating'] = pd.Categorical(df['rating'], categories=categories, ordered=True)

    # ⭐️ 极其重要：标准化所有的 X 和 PC！
    # 如果不标准化，带有随机斜率的 MCMC 采样会因为量纲差异直接崩溃发散
    print("📏 正在对 28 个高维特征进行严格的 Z-score 标准化...")
    for feat in ALL_14_FEATURES + PC_COLS:
        df[feat] = (df[feat] - df[feat].mean()) / df[feat].std()

    print("\n================ [阶段 2] 终极模型循环炼丹 ================")
    # 构造固定效应部分的字符串: "feat1 + feat2 + ... + PC1 + PC2 + ..."
    fixed_x_part = " + ".join(ALL_14_FEATURES)
    pc_part = " + ".join(PC_COLS)
    all_fixed_effects = f"{fixed_x_part} + {pc_part}"

    heterogeneity_scores = []

    for i, target_feat in enumerate(TARGET_FEATURES):
        loop_start = time.time()
        print(f"\n" + "=" * 40)
        print(f"▶️ [{i + 1}/{len(TARGET_FEATURES)}] 正在深度挖掘特征异质性: 【{target_feat}】")
        print("=" * 40)

        # 👑 你的终极公式
        # rating ~ 1 + (14X) + (14PC) + (1|rater) + (1|image) + (0+target|rater)
        formula = f"rating ~ {all_fixed_effects} + (1|rater) + (1|image) + (0+{target_feat}|rater)"

        print(f"📜 注入公式: {formula[:80]} ... (省略中间26个特征) ... + (0+{target_feat}|rater)")

        # 构建并拟合模型
        model = bmb.Model(formula, data=df, family="cumulative")

        # init="adapt_diag" 能极大增加超大模型初始化的稳定性
        results = model.fit(draws=draws, tune=tune, chains=4, target_accept=0.95, init="adapt_diag")

        # 封装时间胶囊
        nc_save_path = os.path.join(SAVE_DIR, f"Ultimate_Heterogeneity_{target_feat}.nc")
        results.to_netcdf(nc_save_path)
        print(f"📦 专属时间胶囊已封存: {nc_save_path}")

        # --- 精准提取随机斜率的方差 (异质性 σ) ---
        summary_df = az.summary(results)
        sigma_row_name = f"{target_feat}|rater_sigma"

        if sigma_row_name in summary_df.index:
            random_slope_sd = summary_df.loc[sigma_row_name, 'mean']
            hdi_3 = summary_df.loc[sigma_row_name, 'hdi_3%']
            hdi_97 = summary_df.loc[sigma_row_name, 'hdi_97%']

            heterogeneity_scores.append({
                'Feature': target_feat,
                'Heterogeneity_SD (σ)': random_slope_sd,
                'HDI_3%': hdi_3,
                'HDI_97%': hdi_97
            })
            print(
                f"🎯 成功提取！【{target_feat}】 纯净异质性指数 (SD): {random_slope_sd:.3f} [HDI: {hdi_3:.3f}, {hdi_97:.3f}]")
        else:
            print(f"⚠️ 警告: 在摘要中未找到 '{sigma_row_name}'！这可能是由于某些版本的 Bambi 命名规则不同。")
            print("当前存在的随机效应参数:", [idx for idx in summary_df.index if 'sigma' in idx or 'rater' in idx])

        # 🧹 极限内存清理 (极其重要！)
        # 超大模型循环极易引发内存泄漏，必须强制释放
        del model
        del results
        gc.collect()

        loop_end = time.time()
        print(f"⏳ 击破该特征耗时: {(loop_end - loop_start) / 60:.1f} 分钟")

    print("\n================ [阶段 3] 榜单出炉 ================")
    if heterogeneity_scores:
        df_het = pd.DataFrame(heterogeneity_scores)
        # 按异质性分歧度，从大到小降序排列
        df_het = df_het.sort_values(by='Heterogeneity_SD (σ)', ascending=False).reset_index(drop=True)

        print("\n🏆 【面部美学特征：人群审美分歧度（异质性）最终排行榜】")
        print("(* 已严苛控制 14大五官几何主效应 & 14大全局深度轮廓特征 *)")
        print("-" * 75)
        print(df_het.to_string(index=True))
        print("-" * 75)

        # 保存为 CSV 供制表使用
        csv_path = os.path.join(SAVE_DIR, "Ultimate_Heterogeneity_Leaderboard.csv")
        df_het.to_csv(csv_path, index=False)
        print(f"\n💾 终极排行榜已永久保存至: {csv_path}")

    total_time = time.time() - start_time
    print(f"\n🎉 全部任务圆满结束！总耗时: {total_time / 60:.1f} 分钟。")


if __name__ == "__main__":
    from multiprocessing import freeze_support

    # Windows 环境下运行多进程 MCMC 必须保留这一行
    freeze_support()
    run_ultimate_models()