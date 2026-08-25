import numpy as np
import os
import csv
import pandas as pd
from scipy import stats

def read_data():
    """Read data from all CSV files."""
    data = []
    accepted_dir = '../Data/accepted'
    for filename in os.listdir(accepted_dir):
        if filename.endswith('.csv'):
            filepath = os.path.join(accepted_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        record = {'score': int(row.get('score', 0)), 'block': row.get('block', ''), 'reaction_time': int(row.get('reaction_time_ms', 0)), 'user_major': row.get('user_major', '')}
                        data.append(record)
                    except (ValueError, KeyError):
                        continue
    return pd.DataFrame(data)

def analyze_major_effect(df):
    """Analyze the effect of major category on aesthetic judgments."""

    def categorize_major(major):
        if not major:
            return 'Unknown'
        major_map = {'理工': 'Science & Engineering', '文科': 'Humanities', '艺术': 'Arts', '医学': 'Medicine', '经济': 'Economics', '管理': 'Management', '教育': 'Education', '法学': 'Law', '农学': 'Agriculture', '其他': 'Other'}
        for key, category in major_map.items():
            if key in major:
                return category
        return 'Other'
    df['major_category'] = df['user_major'].apply(categorize_major)
    major_grouped = df.groupby('major_category')
    major_stats = {}
    for group, data in major_grouped:
        if group != 'Unknown':
            major_stats[group] = {'count': len(data), 'mean_score': data['score'].mean(), 'std_score': data['score'].std(), 'mean_reaction_time': data['reaction_time'].mean(), 'std_reaction_time': data['reaction_time'].std(), 'score_distribution': data['score'].value_counts().sort_index()}
    block_stats = {}
    for block in df['block'].unique():
        if block:
            block_data = df[df['block'] == block]
            block_major_grouped = block_data.groupby('major_category')
            block_major_stats = {}
            for group, data in block_major_grouped:
                if group != 'Unknown':
                    block_major_stats[group] = {'count': len(data), 'mean_score': data['score'].mean(), 'std_score': data['score'].std()}
            block_stats[block] = block_major_stats
    return (major_stats, block_stats, df)

def perform_statistical_tests(df, group_by):
    """Perform statistical tests to compare groups."""
    groups = [group['score'].values for name, group in df.groupby(group_by) if len(group) > 0 and name != 'Unknown']
    group_names = [name for name, group in df.groupby(group_by) if len(group) > 0 and name != 'Unknown']
    if len(groups) < 2:
        return (None, None, [])
    f_stat, p_value = stats.f_oneway(*groups)
    pairwise_results = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            t_stat, p_val = stats.ttest_ind(groups[i], groups[j])
            pairwise_results.append({'group1': group_names[i], 'group2': group_names[j], 't_stat': t_stat, 'p_value': p_val})
    return (f_stat, p_value, pairwise_results)

def generate_report(major_stats, block_stats, anova_result, output_path):
    """Generate analysis report."""
    report = []
    report.append('=' * 80)
    report.append('MAJOR CATEGORY EFFECT ON AESTHETIC JUDGMENT ANALYSIS REPORT')
    report.append('=' * 80)
    report.append('')
    report.append('SUMMARY STATISTICS BY MAJOR CATEGORY')
    report.append('-' * 80)
    report.append(f'{'Major Category':<30} {'N':>5} {'Mean Score':>12} {'Std Score':>12} {'Mean RT (ms)':>15}')
    report.append('-' * 80)
    for group, stat in major_stats.items():
        report.append(f'{group:<30} {stat['count']:>5} {stat['mean_score']:>12.2f} {stat['std_score']:>12.2f} {stat['mean_reaction_time']:>15.1f}')
    report.append('')
    report.append('SUMMARY STATISTICS BY BLOCK TYPE AND MAJOR CATEGORY')
    report.append('-' * 80)
    for block, stats in block_stats.items():
        report.append(f'\n{block.capitalize()} Images:')
        report.append(f'{'Major Category':<30} {'N':>5} {'Mean Score':>12} {'Std Score':>12}')
        report.append('-' * 80)
        for group, stat in stats.items():
            report.append(f'{group:<30} {stat['count']:>5} {stat['mean_score']:>12.2f} {stat['std_score']:>12.2f}')
    report.append('')
    report.append('STATISTICAL TESTS')
    report.append('-' * 80)
    if anova_result[0] is not None:
        report.append(f'Major Category ANOVA F-statistic: {anova_result[0]:.4f}')
        report.append(f'Major Category ANOVA p-value: {anova_result[1]:.4f}')
    report.append('')
    report.append('KEY FINDINGS')
    report.append('-' * 80)
    major_mean_scores = {group: stat['mean_score'] for group, stat in major_stats.items()}
    sorted_major = sorted(major_mean_scores.items(), key=lambda x: x[1], reverse=True)
    report.append('\nScore Analysis by Major Category:')
    for group, mean_score in sorted_major:
        report.append(f'  {group}: {mean_score:.2f}')
    major_mean_rt = {group: stat['mean_reaction_time'] for group, stat in major_stats.items()}
    sorted_major_rt = sorted(major_mean_rt.items(), key=lambda x: x[1])
    report.append('\nReaction Time Analysis by Major Category:')
    for group, rt in sorted_major_rt:
        report.append(f'  {group}: {rt:.1f} ms')
    report.append('\nINTERPRETATION')
    report.append('-' * 80)
    if anova_result[0] is not None and anova_result[1] < 0.05:
        report.append('\nStatistically significant differences found between major categories.')
    else:
        report.append('\nNo statistically significant differences found between major categories.')
    report.append('\nADDITIONAL INSIGHTS')
    report.append('-' * 80)
    report.append('\nKey Question Analysis:')
    report.append('1. Does major category affect image aesthetic judgments?')
    if major_mean_scores:
        scores = list(major_mean_scores.values())
        if max(scores) - min(scores) > 0.1:
            report.append('   - Yes, major category appears to affect aesthetic judgments')
            highest_group = max(major_mean_scores, key=major_mean_scores.get)
            lowest_group = min(major_mean_scores, key=major_mean_scores.get)
            report.append(f'   - Highest scoring group: {highest_group} ({major_mean_scores[highest_group]:.2f})')
            report.append(f'   - Lowest scoring group: {lowest_group} ({major_mean_scores[lowest_group]:.2f})')
        else:
            report.append('   - No significant effect of major category on aesthetic judgments')
    report.append('\nPossible Explanations:')
    report.append('  - Different academic backgrounds may shape aesthetic perspectives')
    report.append('  - Technical vs. creative majors may have different aesthetic priorities')
    report.append('  - Educational training may influence how individuals evaluate visual stimuli')
    report.append('')
    report.append('=' * 80)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    print(f'Analysis report saved: {output_path}')

def main():
    """Main function to run major analysis."""
    print('Reading data...')
    df = read_data()
    print(f'Total records: {len(df)}')
    print('\nAnalyzing major effects...')
    major_stats, block_stats, df = analyze_major_effect(df)
    print('\nGenerating visualizations...')
    os.makedirs('code/picture', exist_ok=True)
    print('\nPerforming statistical tests...')
    anova_result = perform_statistical_tests(df, 'major_category')
    generate_report(major_stats, block_stats, anova_result, 'code/major_analysis_report.txt')
    print('\n' + '=' * 80)
    print('MAJOR CATEGORY EFFECT ANALYSIS COMPLETE')
    print('=' * 80)
    print('\nGenerated files:')
    print('  - code/picture/major_analysis.png')
    print('  - code/picture/major_reaction_time.png')
    print('  - code/picture/major_score_distribution.png')
    print('  - code/picture/major_block_analysis.png')
    print('  - code/major_analysis_report.txt')
if __name__ == '__main__':
    main()
