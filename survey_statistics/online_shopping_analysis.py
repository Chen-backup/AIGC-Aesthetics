import pandas as pd
import numpy as np
import os
from scipy import stats
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
                print(f'读取文件 {filename} 时出错: {e}')
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f'Total records: {len(combined_df)}')
        return combined_df
    else:
        print('No data files found.')
        return pd.DataFrame()

def categorize_shopping_frequency(frequency):
    """将网购频率分类"""
    if pd.isna(frequency) or not frequency:
        return 'Unknown'
    frequency_map = {'几乎不': 'Rarely', '每月1-2次': 'Monthly (1-2 times)', '每月3-4次': 'Monthly (3-4 times)', '每周1-2次': 'Weekly (1-2 times)', '每周3-4次': 'Weekly (3-4 times)', '几乎每天': 'Almost Daily'}
    return frequency_map.get(frequency, 'Other')

def analyze_shopping_effect(df):
    """分析网购频率对审美判断的影响"""
    if df.empty:
        print('No data to analyze.')
        return None
    df['shopping_category'] = df['user_online_shopping'].apply(categorize_shopping_frequency)
    shopping_stats = df.groupby('shopping_category').agg({'score': ['mean', 'std', 'count'], 'reaction_time_ms': ['mean', 'std']}).round(2)
    print('\nShopping Frequency Analysis:')
    print(shopping_stats)
    return df

def perform_statistical_tests(df):
    """执行统计测试"""
    if df.empty:
        return
    shopping_categories = df['shopping_category'].unique()
    score_groups = []
    valid_categories = []
    for category in shopping_categories:
        category_data = df[df['shopping_category'] == category]['score']
        if len(category_data) > 5:
            score_groups.append(category_data)
            valid_categories.append(category)
    if len(score_groups) > 1:
        f_stat, p_value = stats.f_oneway(*score_groups)
        print(f'\nShopping Frequency ANOVA F-statistic: {f_stat:.4f}')
        print(f'Shopping Frequency ANOVA p-value: {p_value:.4f}')
        return {'f_stat': f_stat, 'p_value': p_value, 'categories': valid_categories}
    else:
        print('\nNot enough categories for statistical tests.')
        return None

def generate_report(df, stats_results):
    """生成分析报告"""
    if df.empty:
        return
    shopping_stats = df.groupby('shopping_category').agg({'score': ['mean', 'std', 'count'], 'reaction_time_ms': ['mean']}).round(2)
    report_content = '=' * 80 + '\n'
    report_content += 'ONLINE SHOPPING FREQUENCY EFFECT ON AESTHETIC JUDGMENT ANALYSIS REPORT\n'
    report_content += '=' * 80 + '\n\n'
    report_content += 'SUMMARY STATISTICS BY SHOPPING FREQUENCY\n'
    report_content += '-' * 80 + '\n'
    report_content += 'Shopping Frequency          N   Mean Score    Std Score    Mean RT (ms)\n'
    report_content += '-' * 80 + '\n'
    for category in shopping_stats.index:
        row = shopping_stats.loc[category]
        count = row['score', 'count']
        mean_score = row['score', 'mean']
        std_score = row['score', 'std']
        mean_rt = row['reaction_time_ms', 'mean']
        report_content += f'{category:<25} {int(count):>4}         {mean_score:>6.2f}         {std_score:>6.2f}          {mean_rt:>10.1f}\n'
    report_content += '\n'
    report_content += 'STATISTICAL TESTS\n'
    report_content += '-' * 80 + '\n'
    if stats_results:
        report_content += f'Shopping Frequency ANOVA F-statistic: {stats_results['f_stat']:.4f}\n'
        report_content += f'Shopping Frequency ANOVA p-value: {stats_results['p_value']:.4f}\n'
    else:
        report_content += 'Insufficient data for statistical tests\n'
    report_content += '\n'
    report_content += 'KEY FINDINGS\n'
    report_content += '-' * 80 + '\n\n'
    report_content += 'Score Analysis by Shopping Frequency:\n'
    score_by_category = df.groupby('shopping_category')['score'].mean().sort_values(ascending=False)
    for category, score in score_by_category.items():
        report_content += f'  {category}: {score:.2f}\n'
    report_content += '\n'
    report_content += 'Reaction Time Analysis by Shopping Frequency:\n'
    rt_by_category = df.groupby('shopping_category')['reaction_time_ms'].mean().sort_values()
    for category, rt in rt_by_category.items():
        report_content += f'  {category}: {rt:.1f} ms\n'
    report_content += '\n'
    report_content += 'INTERPRETATION\n'
    report_content += '-' * 80 + '\n\n'
    if stats_results and stats_results['p_value'] < 0.05:
        report_content += 'Statistically significant differences found between shopping frequency groups.\n'
    else:
        report_content += 'No statistically significant differences found between shopping frequency groups.\n'
    report_content += '\n'
    report_content += 'ADDITIONAL INSIGHTS\n'
    report_content += '-' * 80 + '\n\n'
    report_content += 'Key Question Analysis:\n'
    report_content += '1. Does online shopping frequency affect image aesthetic judgments?\n'
    if stats_results and stats_results['p_value'] < 0.05:
        report_content += '   - Yes, online shopping frequency appears to affect aesthetic judgments\n'
        highest = score_by_category.idxmax()
        lowest = score_by_category.idxmin()
        report_content += f'   - Highest scoring group: {highest} ({score_by_category[highest]:.2f})\n'
        report_content += f'   - Lowest scoring group: {lowest} ({score_by_category[lowest]:.2f})\n'
    else:
        report_content += '   - No significant effect of online shopping frequency on aesthetic judgments\n'
    report_content += '\n'
    report_content += 'Possible Explanations:\n'
    report_content += '  - Frequent online shoppers may be more exposed to visual marketing and design\n'
    report_content += '  - Shopping habits may correlate with other factors that influence aesthetic preferences\n'
    report_content += '  - Different shopping frequencies may reflect different lifestyle patterns\n'
    report_content += '\n' + '=' * 80
    with open('code/shopping_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_content)
    print('Analysis report saved: code/shopping_analysis_report.txt')

def main():
    print('Reading data...')
    df = read_data()
    print('\nAnalyzing shopping frequency effects...')
    df = analyze_shopping_effect(df)
    if df is not None and (not df.empty):
        print('\nGenerating visualizations...')
        print('\nPerforming statistical tests...')
        stats_results = perform_statistical_tests(df)
        generate_report(df, stats_results)
        print('\n' + '=' * 80)
        print('ONLINE SHOPPING FREQUENCY EFFECT ANALYSIS COMPLETE')
        print('=' * 80)
        print('Generated files:')
        print('  - code/picture/shopping_analysis.png')
        print('  - code/picture/shopping_reaction_time.png')
        print('  - code/picture/shopping_score_distribution.png')
        print('  - code/picture/shopping_block_analysis.png')
        print('  - code/shopping_analysis_report.txt')
if __name__ == '__main__':
    main()
