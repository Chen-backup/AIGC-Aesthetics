import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 确保图片保存目录存在
os.makedirs('code/picture', exist_ok=True)

def read_data():
    """读取所有CSV文件并合并为一个DataFrame"""
    data_dir = '../Data/accepted'
    all_data = []
    
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            file_path = os.path.join(data_dir, filename)
            try:
                df = pd.read_csv(file_path)
                all_data.append(df)
            except Exception as e:
                print(f"读取文件 {filename} 时出错: {e}")
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"Total records: {len(combined_df)}")
        return combined_df
    else:
        print("No data files found.")
        return pd.DataFrame()

def categorize_subscription(subscription):
    """将会员/订阅情况分类"""
    if pd.isna(subscription) or not subscription or subscription.strip() == '':
        return 'No Subscription'
    else:
        return 'With Subscription'

def analyze_subscription_effect(df):
    """分析会员/订阅情况对审美判断的影响"""
    if df.empty:
        print("No data to analyze.")
        return None
    
    # 分类会员/订阅情况
    df['subscription_category'] = df['user_subscriptions'].apply(categorize_subscription)
    
    # 按会员/订阅情况分组分析
    subscription_stats = df.groupby('subscription_category').agg({
        'score': ['mean', 'std', 'count'],
        'reaction_time_ms': ['mean', 'std']
    }).round(2)
    
    print("\nSubscription Analysis:")
    print(subscription_stats)
    
    return df

def plot_subscription_analysis(df):
    """绘制会员/订阅情况分析图表"""
    if df.empty:
        return
    
    # 评分分析
    plt.figure(figsize=(10, 6))
    subscription_score = df.groupby('subscription_category')['score'].mean()
    subscription_score.plot(kind='bar', color=['lightcoral', 'skyblue'])
    plt.title('平均评分 by 会员/订阅情况')
    plt.xlabel('会员/订阅情况')
    plt.ylabel('平均评分')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('code/picture/subscription_analysis.png')
    print("Subscription analysis plot saved: code/picture/subscription_analysis.png")
    
    # 反应时间分析
    plt.figure(figsize=(10, 6))
    subscription_rt = df.groupby('subscription_category')['reaction_time_ms'].mean()
    subscription_rt.plot(kind='bar', color=['lightgreen', 'lightblue'])
    plt.title('平均反应时间 by 会员/订阅情况')
    plt.xlabel('会员/订阅情况')
    plt.ylabel('平均反应时间 (ms)')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('code/picture/subscription_reaction_time.png')
    print("Reaction time comparison plot saved: code/picture/subscription_reaction_time.png")
    
    # 评分分布
    plt.figure(figsize=(14, 8))
    subscription_categories = df['subscription_category'].unique()
    for category in subscription_categories:
        category_data = df[df['subscription_category'] == category]['score']
        if len(category_data) > 0:
            plt.hist(category_data, bins=7, alpha=0.5, label=category)
    plt.title('评分分布 by 会员/订阅情况')
    plt.xlabel('评分')
    plt.ylabel('频数')
    plt.legend()
    plt.tight_layout()
    plt.savefig('code/picture/subscription_score_distribution.png')
    print("Score distribution plot saved: code/picture/subscription_score_distribution.png")
    
    # 按图像类型分析
    plt.figure(figsize=(14, 8))
    block_subscription = df.groupby(['block', 'subscription_category'])['score'].mean().unstack()
    block_subscription.plot(kind='bar', ax=plt.gca())
    plt.title('不同图像类型的评分 by 会员/订阅情况')
    plt.xlabel('图像类型')
    plt.ylabel('平均评分')
    plt.xticks(rotation=45)
    plt.legend(title='会员/订阅情况')
    plt.tight_layout()
    plt.savefig('code/picture/subscription_block_analysis.png')
    print("Block subscription analysis plot saved: code/picture/subscription_block_analysis.png")

def perform_statistical_tests(df):
    """执行统计测试"""
    if df.empty:
        return
    
    # 提取不同会员/订阅情况的评分数据
    subscription_categories = df['subscription_category'].unique()
    if len(subscription_categories) != 2:
        print("\nNot enough categories for statistical tests.")
        return None
    
    # 独立样本t检验
    group1 = df[df['subscription_category'] == 'No Subscription']['score']
    group2 = df[df['subscription_category'] == 'With Subscription']['score']
    
    t_stat, p_value = stats.ttest_ind(group1, group2)
    print(f"\nSubscription T-test t-statistic: {t_stat:.4f}")
    print(f"Subscription T-test p-value: {p_value:.4f}")
    
    return {
        't_stat': t_stat,
        'p_value': p_value,
        'categories': subscription_categories
    }

def generate_report(df, stats_results):
    """生成分析报告"""
    if df.empty:
        return
    
    # 计算基本统计信息
    subscription_stats = df.groupby('subscription_category').agg({
        'score': ['mean', 'std', 'count'],
        'reaction_time_ms': ['mean']
    }).round(2)
    
    # 准备报告内容
    report_content = "=" * 80 + "\n"
    report_content += "SUBSCRIPTION STATUS EFFECT ON AESTHETIC JUDGMENT ANALYSIS REPORT\n"
    report_content += "=" * 80 + "\n\n"
    
    report_content += "SUMMARY STATISTICS BY SUBSCRIPTION STATUS\n"
    report_content += "-" * 80 + "\n"
    report_content += "Subscription Status        N   Mean Score    Std Score    Mean RT (ms)\n"
    report_content += "-" * 80 + "\n"
    
    for category in subscription_stats.index:
        row = subscription_stats.loc[category]
        count = row[('score', 'count')]
        mean_score = row[('score', 'mean')]
        std_score = row[('score', 'std')]
        mean_rt = row[('reaction_time_ms', 'mean')]
        report_content += f"{category:<25} {int(count):>4}         {mean_score:>6.2f}         {std_score:>6.2f}          {mean_rt:>10.1f}\n"
    
    report_content += "\n"
    report_content += "STATISTICAL TESTS\n"
    report_content += "-" * 80 + "\n"
    
    if stats_results:
        report_content += f"Subscription T-test t-statistic: {stats_results['t_stat']:.4f}\n"
        report_content += f"Subscription T-test p-value: {stats_results['p_value']:.4f}\n"
    else:
        report_content += "Insufficient data for statistical tests\n"
    
    report_content += "\n"
    report_content += "KEY FINDINGS\n"
    report_content += "-" * 80 + "\n\n"
    
    # 评分分析
    report_content += "Score Analysis by Subscription Status:\n"
    score_by_category = df.groupby('subscription_category')['score'].mean().sort_values(ascending=False)
    for category, score in score_by_category.items():
        report_content += f"  {category}: {score:.2f}\n"
    
    report_content += "\n"
    report_content += "Reaction Time Analysis by Subscription Status:\n"
    rt_by_category = df.groupby('subscription_category')['reaction_time_ms'].mean().sort_values()
    for category, rt in rt_by_category.items():
        report_content += f"  {category}: {rt:.1f} ms\n"
    
    report_content += "\n"
    report_content += "INTERPRETATION\n"
    report_content += "-" * 80 + "\n\n"
    
    if stats_results and stats_results['p_value'] < 0.05:
        report_content += "Statistically significant differences found between subscription status groups.\n"
    else:
        report_content += "No statistically significant differences found between subscription status groups.\n"
    
    report_content += "\n"
    report_content += "ADDITIONAL INSIGHTS\n"
    report_content += "-" * 80 + "\n\n"
    
    report_content += "Key Question Analysis:\n"
    report_content += "1. Does subscription status affect image aesthetic judgments?\n"
    if stats_results and stats_results['p_value'] < 0.05:
        report_content += "   - Yes, subscription status appears to affect aesthetic judgments\n"
        highest = score_by_category.idxmax()
        lowest = score_by_category.idxmin()
        report_content += f"   - Highest scoring group: {highest} ({score_by_category[highest]:.2f})\n"
        report_content += f"   - Lowest scoring group: {lowest} ({score_by_category[lowest]:.2f})\n"
    else:
        report_content += "   - No significant effect of subscription status on aesthetic judgments\n"
    
    report_content += "\n"
    report_content += "Possible Explanations:\n"
    report_content += "  - Subscribers may have access to higher quality content, influencing their aesthetic standards\n"
    report_content += "  - Paying for subscriptions may indicate a higher value placed on content quality\n"
    report_content += "  - Subscription status may correlate with other factors that influence aesthetic preferences\n"
    
    report_content += "\n" + "=" * 80
    
    # 保存报告
    with open('code/subscription_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print("Analysis report saved: code/subscription_analysis_report.txt")

def main():
    print("Reading data...")
    df = read_data()
    
    print("\nAnalyzing subscription effects...")
    df = analyze_subscription_effect(df)
    
    if df is not None and not df.empty:
        print("\nGenerating visualizations...")
        plot_subscription_analysis(df)
        
        print("\nPerforming statistical tests...")
        stats_results = perform_statistical_tests(df)
        
        generate_report(df, stats_results)
        
        print("\n" + "=" * 80)
        print("SUBSCRIPTION STATUS EFFECT ANALYSIS COMPLETE")
        print("=" * 80)
        print("Generated files:")
        print("  - code/picture/subscription_analysis.png")
        print("  - code/picture/subscription_reaction_time.png")
        print("  - code/picture/subscription_score_distribution.png")
        print("  - code/picture/subscription_block_analysis.png")
        print("  - code/subscription_analysis_report.txt")

if __name__ == "__main__":
    main()
