import arviz as az
import pandas as pd
import os

print("\n" + "★" * 60)
print("📦 启动【14维偏好提取器：从 .nc 时间胶囊生成 CSV】")
print("★" * 60)

# ==========================================
# 1. 路径配置
# ==========================================
DIR_NC = "BYS_Ultimate_Heterogeneity_Results"
SAVE_DIR = "BYS_Clustering_Results_14D"
os.makedirs(SAVE_DIR, exist_ok=True)

# 14 个可解释特征
TARGET_FEATURES = [
    'face_hw_ratio', 'eye_face_w_ratio', 'mouth_face_w_ratio', 'three_courts_balance',
    'upper_lower_ratio', 'eye_y_ratio', 'total_symmetry', 'le_nose_re_angle',
    'mouth_nose_ratio', 'face_brightness', 'face_contrast', 'face_clarity',
    'saturation', 'edge_density'
]

# ==========================================
# 2. 循环提取
# ==========================================
print("\n⏳ 正在潜入 14 个时间胶囊，提取所有评委的潜意识偏好 (BLUPs)...")
rater_slopes_list = []

for feat in TARGET_FEATURES:
    nc_path = os.path.join(DIR_NC, f"Ultimate_Heterogeneity_{feat}.nc")
    if not os.path.exists(nc_path):
        print(f"❌ 警告：找不到文件 {nc_path}！请确认它存在。")
        continue

    print(f"   -> 正在提取特征: {feat}")
    try:
        # 打开时间胶囊
        trace = az.from_netcdf(nc_path)
        var_name = f"{feat}|rater"

        # 将 chain 和 draw 这两万次采样压缩成一个均值
        mean_slopes = trace.posterior[var_name].mean(dim=["chain", "draw"])

        # 转换成 pandas 列
        s = mean_slopes.to_series()
        s.name = feat
        rater_slopes_list.append(s)

        # 关上门，清理内存
        del trace
    except KeyError:
        print(f"⚠️ 在 {feat} 中找不到变量 '{var_name}'！")
    except Exception as e:
        print(f"⚠️ 读取 {feat} 发生未知错误: {e}")

# ==========================================
# 3. 拼接与保存
# ==========================================
if rater_slopes_list:
    # 拼接所有的特征列 (1300 x 14)
    df_preferences = pd.concat(rater_slopes_list, axis=1)

    # 剔除存在缺失值的数据行，保证数据纯净
    df_preferences = df_preferences.dropna()

    # 保存！
    csv_path = os.path.join(SAVE_DIR, "Rater_14D_Preferences.csv")
    df_preferences.to_csv(csv_path)

    print(f"\n✅ 成功！提取了 {len(df_preferences)} 名有效评委的 14 维审美偏好。")
    print(f"💾 文件已完美钉死并保存至: {csv_path}")
    print("\n🚀 现在，你可以放心大胆地去运行 find_optimal_k.py (最优 K 值诊断) 了！")
else:
    print("\n❌ 提取失败，未能生成任何数据。请检查 BYS_Ultimate_Heterogeneity_Results 文件夹中是否有 .nc 文件。")