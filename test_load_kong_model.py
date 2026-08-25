import os
import pickle
import arviz as az
import bambi as bmb  # 必须导入 bambi，否则 pickle 无法识别其内部对象结构


def test_model_artifacts():
    save_dir = "BYS_kong_2_result"
    nc_path = os.path.join(save_dir, "null_model_trace.nc")
    pkl_path = os.path.join(save_dir, "null_model_object.pkl")

    print("================ 🔍 开始验证模型存档 ================\n")

    # -----------------------------------------
    # 测试 1: 验证 MCMC 采样轨迹 (.nc 文件)
    # -----------------------------------------
    print(f"👉 正在检查后验采样时间胶囊: {nc_path}")
    if os.path.exists(nc_path):
        try:
            # 尝试加载 NetCDF 文件
            results = az.from_netcdf(nc_path)
            print("   ✅ 成功读取 .nc 文件！")

            # 提取并验证数据维度
            chains = results.posterior.chain.size
            draws = results.posterior.draw.size
            print(f"   📊 数据验证: 包含 {chains} 条马尔可夫链，每条链 {draws} 次采样。")

            # 尝试提取一个核心参数，证明数据确实在里面
            summary = az.summary(results, var_names=["1|image_sigma", "1|rater_sigma"])
            print("   📝 核心方差参数快照:")
            print(summary[['mean', 'sd', 'r_hat']])

        except Exception as e:
            print(f"   ❌ 读取 .nc 文件时发生错误:\n{e}")
    else:
        print(f"   ❌ 找不到文件: {nc_path}")

    print("\n------------------------------------------------------\n")

    # -----------------------------------------
    # 测试 2: 验证模型结构对象 (.pkl 文件)
    # -----------------------------------------
    print(f"👉 正在检查模型结构对象: {pkl_path}")
    if os.path.exists(pkl_path):
        try:
            # 尝试加载 Pickle 文件
            with open(pkl_path, "rb") as f:
                model = pickle.load(f)
            print("   ✅ 成功读取 .pkl 文件！")

            # 提取并验证模型属性
            print(f"   🧠 模型公式: {model.formula}")
            print(f"   🧠 响应分布族 (Family): {model.family.name}")
            print(f"   🧠 响应变量连接函数 (Link): {model.family.link.name}")

        except Exception as e:
            print(f"   ❌ 读取 .pkl 文件时发生错误:\n{e}")
    else:
        print(f"   ❌ 找不到文件: {pkl_path}")

    print("\n================ 🎉 验证程序执行完毕 ================")
    print("💡 结论：如果上方两项均显示 ✅，说明你的存档机制绝对可靠，可以放心开启全量运行！")


if __name__ == "__main__":
    test_model_artifacts()