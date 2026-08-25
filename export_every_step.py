import arviz as az
import pandas as pd
import numpy as np
import os


def export_trace_to_csv_npy():
    print("================ 正在打开时间胶囊 ================")
    nc_path = "BYS_kong_2_result/null_model_trace.nc"

    if not os.path.exists(nc_path):
        print(f"❌ 找不到文件: {nc_path}")
        return

    results = az.from_netcdf(nc_path)
    print("✅ 成功加载 .nc 模型数据！")

    print("\n================ 正在展平高维数据 ================")
    # 核心魔法：把多维的后验采样数据展平为一个巨大的 Pandas DataFrame
    # 里面包含了 chain(链号), draw(步数), 以及每一个参数在这一步的值
    trace_df = results.posterior.to_dataframe().reset_index()

    print(f"✅ 数据展平成功！数据表维度: {trace_df.shape}")
    print(f"   (行数 = 链数 × 每链步数 × 图片数 × 评分人数)")

    print("\n================ 正在导出文件 ================")

    # 1. 导出为 CSV (可以用 Excel 或普通代码读取，但文件体积可能很大)
    csv_path = "BYS_kong_2_result/Null_Model_EveryStep.csv"
    print(f"👉 正在保存 CSV 到 {csv_path} (可能需要几分钟)...")
    trace_df.to_csv(csv_path, index=False)
    print("   ✅ CSV 保存完成！")

    # 2. 导出为 NumPy (.npy) 格式 (适合 Python 深度学习读取，体积更小，速度极快)
    npy_path = "BYS_kong_2_result/Null_Model_EveryStep.npy"
    print(f"👉 正在保存 NPY 到 {npy_path}...")
    # 将 DataFrame 转换为结构化 NumPy 数组并保存
    np.save(npy_path, trace_df.to_records(index=False))
    print("   ✅ NPY 保存完成！")

    print("\n🎉 全部导出成功！你现在拥有了 MCMC 采样每一步的绝对原始记录。")


if __name__ == '__main__':
    export_trace_to_csv_npy()