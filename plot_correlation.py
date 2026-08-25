import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_feature_correlation():
    print("================ 1. 读取面部特征数据 ================")
    # 直接读取你的 14维特征数据表
    df_geom = pd.read_csv("interpretable_face_features.csv")

    # 提取我们关心的这 4 个核心特征
    features = ['face_hw_ratio', 'eye_face_w_ratio', 'three_courts_balance', 'mouth_nose_ratio']

    # 检查列名是否存在，存在则提取
    available_features = [f for f in features if f in df_geom.columns]
    df_sub = df_geom[available_features].dropna()

    print(f"✅ 成功提取 {len(df_sub)} 张人脸的特征数据。")

    print("\n================ 2. 计算并绘制 Spearman 相关系数矩阵 ================")
    # 面部特征通常不完全是正态分布，使用 spearman 秩相关更严谨
    corr_matrix = df_sub.corr(method='spearman')

    plt.figure(figsize=(8, 6))
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    sns.set_context("paper", font_scale=1.2)

    # 绘制热力图 (红蓝配色：红色正相关，蓝色负相关，颜色越浅越不相关)
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0,
                fmt='.2f', square=True, linewidths=.5,
                cbar_kws={"shrink": .8, "label": "Spearman Correlation"})

    plt.title('Correlation Matrix of the 4 Key Geometric Features', weight='bold', pad=15)

    # 调整 x 轴标签倾斜度，防止字叠在一起
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    save_path = "Feature_Correlation_Heatmap.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"✅ 热力图已生成！请查看: {os.path.abspath(save_path)}")


if __name__ == "__main__":
    plot_feature_correlation()