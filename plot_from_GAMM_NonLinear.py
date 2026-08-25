import pandas as pd
import bambi as bmb
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


def plot_from_saved_model():
    print("================ 1. 瞬间加载模型灵魂 ================")
    nc_path = os.path.join("BYS_GAMM_NonLinear_Result", "GAMM_model_trace.nc")

    if not os.path.exists(nc_path):
        print(f"❌ 找不到模型文件: {nc_path}")
        print("请确认你已经在上一次全量训练时保存了 .nc 文件！")
        return

    trace = az.from_netcdf(nc_path)
    print("✅ 几个小时的训练心血已在 1 秒内加载到内存！")

    print("\n================ 2. 重塑模型肉身 (数据清洗与空壳构建) ================")
    # 重新读取数据，获取标准化参数（用于画图的 X 轴还原）
    df_ratings = pd.read_excel("ratings_for_bayesian_model.xlsx")
    df_mapping = pd.read_excel("renumber&gender.xlsx")
    df_geom = pd.read_csv("interpretable_face_features.csv")

    geom_merge_key = 'face_id' if 'face_id' in df_geom.columns else 'image_name'
    df_combined = pd.merge(df_mapping[['face_id', 'Number']], df_geom, left_on='face_id', right_on=geom_merge_key,
                           how='inner')
    df = pd.merge(df_ratings, df_combined, left_on='image', right_on='Number', how='inner')

    nonlinear_features = ['face_hw_ratio', 'eye_face_w_ratio', 'three_courts_balance', 'mouth_nose_ratio']
    df = df.dropna(subset=nonlinear_features + ['rating', 'rater', 'image'])

    df['rater'] = df['rater'].astype(str)
    df['image'] = df['image'].astype(str)

    categories = sorted(df['rating'].unique())
    df['rating'] = pd.Categorical(df['rating'], categories=categories, ordered=True)
    rating_cats_numeric = np.array(categories).astype(float)

    # 提取并保存物理边界和标准化参数
    feature_stats = {}
    for feat in nonlinear_features:
        mean_val, std_val = df[feat].mean(), df[feat].std()
        df[feat] = (df[feat] - mean_val) / std_val  # 标准化
        feature_stats[feat] = {
            'mean': mean_val,
            'std': std_val,
            'z_min': df[feat].min(),
            'z_max': df[feat].max()
        }

    # 构建一模一样的空壳公式
    spline_terms = [f"bs({feat}, df=4)" for feat in nonlinear_features]
    formula = f"rating ~ 1 + {' + '.join(spline_terms)} + (1|rater) + (1|image)"

    # ⚠️ 瞬间初始化空模型！(绝对不写 model.fit)
    model = bmb.Model(formula, data=df, family="cumulative")
    print("✅ 模型空壳搭建完毕！")

    print("\n================ 3. 极速预测与渲染期望审美曲线 ================")
    save_dir = "BYS_GAMM_NonLinear_Result"

    plt.figure(figsize=(15, 10))
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    sns.set_context("paper", font_scale=1.2)

    base_rater = df['rater'].iloc[0]
    base_image = df['image'].iloc[0]

    for i, feat in enumerate(nonlinear_features):
        print(f"正在瞬间预测并绘制 {feat} ...")

        # 提取防越界的网格点
        z_min = feature_stats[feat]['z_min']
        z_max = feature_stats[feat]['z_max']
        margin = (z_max - z_min) * 0.01
        x_zscores = np.linspace(z_min + margin, z_max - margin, 100)

        # 构造完美假数据
        dummy_data = {f: np.zeros(100) for f in nonlinear_features}
        dummy_data[feat] = x_zscores
        dummy_data['rater'] = base_rater
        dummy_data['image'] = base_image
        dummy_df = pd.DataFrame(dummy_data)

        # 🎯 核心魔法：直接把 trace 塞进 predict 里进行瞬间预测！
        pred = model.predict(trace, data=dummy_df, kind="response_params", include_group_specific=False, inplace=False)

        # 提取概率矩阵
        pred_values = None
        available_vars = list(pred.posterior.data_vars.keys())
        target_names = ['p', 'rating_response_params', 'rating_probs', 'rating_mean', 'rating']

        for name in target_names:
            if name in available_vars:
                val = pred.posterior[name].values
                if len(val.shape) >= 3:
                    pred_values = val
                    break

        if pred_values is None:
            raise ValueError(f"❌ 找不到预测概率！可用变量: {available_vars}")

        # 计算期望得分 (Expected Score)
        if len(pred_values.shape) == 4:
            expected_scores = np.sum(pred_values * rating_cats_numeric, axis=-1)
        elif len(pred_values.shape) == 3:
            expected_scores = np.sum(pred_values * rating_cats_numeric, axis=-1)

        mean_expected_score = expected_scores.mean(axis=(0, 1))
        lower_bound = np.percentile(expected_scores, 2.5, axis=(0, 1))
        upper_bound = np.percentile(expected_scores, 97.5, axis=(0, 1))

        # 还原物理数值
        x_real = x_zscores * feature_stats[feat]['std'] + feature_stats[feat]['mean']

        # 过滤 NaN 并画图
        valid_mask = ~np.isnan(mean_expected_score)
        if not valid_mask.any():
            print(f"⚠️ 警告: {feat} 的所有预测值均为 NaN，跳过画图。")
            continue

        plt.subplot(2, 2, i + 1)
        plt.plot(x_real[valid_mask], mean_expected_score[valid_mask], color='#d62728', linewidth=3,
                 label='Expected Score')
        plt.fill_between(x_real[valid_mask], lower_bound[valid_mask], upper_bound[valid_mask], color='#d62728',
                         alpha=0.2, label='95% Credible Interval')

        plt.title(f'Non-linear Aesthetic Effect of {feat}', weight='bold', pad=10)
        plt.xlabel(f'Real Values of {feat}', weight='bold')
        plt.ylabel('Expected Aesthetic Score', weight='bold')
        plt.legend(loc='best')
        plt.grid(axis='both', linestyle='--', alpha=0.3)
        sns.despine()

    plt.tight_layout()
    # 另存为一个新名字，或者覆盖原图都可以
    save_path = os.path.join(save_dir, "GAMM_NonLinear_Expected_Curves_Reloaded.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"\n✅ 极速渲染完成！神级曲线已重新生成至: {os.path.abspath(save_path)}")


if __name__ == "__main__":
    # 使用保存的模型预测时，通常不需要多进程支持，但保留无妨
    plot_from_saved_model()