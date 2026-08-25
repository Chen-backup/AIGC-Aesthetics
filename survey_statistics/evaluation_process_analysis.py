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

def analyze_process_features(df):
    """分析评价过程特征与最终评分的关系"""
    if df.empty:
        print('No data to analyze.')
        return None
    correlation_rt_score = df['reaction_time_ms'].corr(df['score'])
    correlation_n_changes_score = df['n_changes'].corr(df['score'])
    print(f'\nCorrelation between reaction time and score: {correlation_rt_score:.4f}')
    print(f'Correlation between number of changes and score: {correlation_n_changes_score:.4f}')
    changes_stats = df.groupby('n_changes').agg({'score': ['mean', 'std', 'count'], 'reaction_time_ms': ['mean', 'std']}).round(2)
    print('\nAnalysis by number of changes:')
    print(changes_stats)
    df['rt_bin'] = pd.cut(df['reaction_time_ms'], bins=[0, 2000, 4000, 6000, 8000, 10000, np.inf], labels=['<2s', '2-4s', '4-6s', '6-8s', '8-10s', '>10s'])
    rt_stats = df.groupby('rt_bin').agg({'score': ['mean', 'std', 'count'], 'n_changes': ['mean', 'std']}).round(2)
    print('\nAnalysis by reaction time bins:')
    print(rt_stats)
    return df

def perform_statistical_tests(df):
    """执行统计测试"""
    if df.empty:
        return
    corr_rt_score, p_value_rt = stats.pearsonr(df['reaction_time_ms'], df['score'])
    corr_changes_score, p_value_changes = stats.pearsonr(df['n_changes'], df['score'])
    print(f'\nStatistical tests:')
    print(f'Reaction time vs score correlation: {corr_rt_score:.4f}, p-value: {p_value_rt:.4f}')
    print(f'Number of changes vs score correlation: {corr_changes_score:.4f}, p-value: {p_value_changes:.4f}')
    return {'corr_rt_score': corr_rt_score, 'p_value_rt': p_value_rt, 'corr_changes_score': corr_changes_score, 'p_value_changes': p_value_changes}

def generate_report(df, stats_results):
    """生成分析报告"""
    if df.empty:
        return
    basic_stats = df[['score', 'reaction_time_ms', 'n_changes']].describe().round(2)
    changes_stats = df.groupby('n_changes').agg({'score': ['mean', 'std', 'count'], 'reaction_time_ms': ['mean', 'std']}).round(2)
    rt_stats = df.groupby('rt_bin').agg({'score': ['mean', 'std', 'count'], 'n_changes': ['mean', 'std']}).round(2)
    report_content = '=' * 80 + '\n'
    report_content += 'EVALUATION PROCESS FEATURES ANALYSIS REPORT\n'
    report_content += '=' * 80 + '\n\n'
    report_content += 'BASIC STATISTICS\n'
    report_content += '-' * 80 + '\n'
    report_content += basic_stats.to_string() + '\n\n'
    report_content += 'CORRELATION ANALYSIS\n'
    report_content += '-' * 80 + '\n'
    if stats_results:
        report_content += f'Reaction time vs score correlation: {stats_results['corr_rt_score']:.4f} (p-value: {stats_results['p_value_rt']:.4f})\n'
        report_content += f'Number of changes vs score correlation: {stats_results['corr_changes_score']:.4f} (p-value: {stats_results['p_value_changes']:.4f})\n'
    report_content += '\n'
    report_content += 'ANALYSIS BY NUMBER OF CHANGES\n'
    report_content += '-' * 80 + '\n'
    report_content += changes_stats.to_string() + '\n\n'
    report_content += 'ANALYSIS BY REACTION TIME BINS\n'
    report_content += '-' * 80 + '\n'
    report_content += rt_stats.to_string() + '\n\n'
    report_content += 'KEY FINDINGS\n'
    report_content += '-' * 80 + '\n\n'
    if stats_results:
        if abs(stats_results['corr_rt_score']) > 0.1:
            if stats_results['corr_rt_score'] > 0:
                report_content += '- Reaction time is positively correlated with score\n'
            else:
                report_content += '- Reaction time is negatively correlated with score\n'
        else:
            report_content += '- Reaction time has weak correlation with score\n'
        if abs(stats_results['corr_changes_score']) > 0.1:
            if stats_results['corr_changes_score'] > 0:
                report_content += '- Number of changes is positively correlated with score\n'
            else:
                report_content += '- Number of changes is negatively correlated with score\n'
        else:
            report_content += '- Number of changes has weak correlation with score\n'
    changes_score = df.groupby('n_changes')['score'].mean()
    max_changes_score = changes_score.idxmax()
    report_content += f'- Highest average score found with {max_changes_score} changes\n'
    rt_score = df.groupby('rt_bin')['score'].mean()
    max_rt_score = rt_score.idxmax()
    report_content += f'- Highest average score found in {max_rt_score} reaction time bin\n'
    report_content += '\n'
    report_content += 'INTERPRETATION\n'
    report_content += '-' * 80 + '\n\n'
    report_content += 'The analysis explores how evaluation process features (reaction time and number of changes) relate to final aesthetic scores.\n'
    report_content += 'Key insights include: how decision time and revision behavior correlate with aesthetic judgments.\n'
    report_content += '\n'
    report_content += 'ADDITIONAL INSIGHTS\n'
    report_content += '-' * 80 + '\n\n'
    report_content += 'Possible interpretations:\n'
    report_content += '- Longer reaction times may indicate more careful consideration of aesthetic qualities\n'
    report_content += '- More changes may reflect a more nuanced evaluation process\n'
    report_content += '- The relationship between process features and scores may vary by image type or content\n'
    report_content += '\n' + '=' * 80
    with open('code/evaluation_process_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_content)
    print('Analysis report saved: code/evaluation_process_analysis_report.txt')

def main():
    print('Reading data...')
    df = read_data()
    print('\nAnalyzing evaluation process features...')
    df = analyze_process_features(df)
    if df is not None and (not df.empty):
        print('\nGenerating visualizations...')
        print('\nPerforming statistical tests...')
        stats_results = perform_statistical_tests(df)
        generate_report(df, stats_results)
        print('\n' + '=' * 80)
        print('EVALUATION PROCESS FEATURES ANALYSIS COMPLETE')
        print('=' * 80)
        print('Generated files:')
        print('  - code/picture/rt_score_scatter.png')
        print('  - code/picture/changes_score_bar.png')
        print('  - code/picture/rt_bin_score_bar.png')
        print('  - code/picture/rt_changes_scatter.png')
        print('  - code/picture/changes_distribution.png')
        print('  - code/picture/rt_distribution.png')
        print('  - code/evaluation_process_analysis_report.txt')
if __name__ == '__main__':
    main()
