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
                        record = {'score': int(row.get('score', 0)), 'block': row.get('block', ''), 'reaction_time': int(row.get('reaction_time_ms', 0)), 'user_photo_exp': row.get('user_photo_exp', '')}
                        data.append(record)
                    except (ValueError, KeyError):
                        continue
    return pd.DataFrame(data)

def analyze_photo_exp_effect(df):
    """Analyze the effect of photography experience on aesthetic judgments."""

    def categorize_photo_exp(exp):
        if not exp:
            return 'Unknown'
        exp_map = {'无': 'No Experience', '手机随拍': 'Mobile Photography', '业余爱好': 'Amateur', '专业摄影师': 'Professional'}
        return exp_map.get(exp, 'Unknown')
    df['photo_exp_category'] = df['user_photo_exp'].apply(categorize_photo_exp)
    photo_exp_grouped = df.groupby('photo_exp_category')
    photo_exp_stats = {}
    for group, data in photo_exp_grouped:
        if group != 'Unknown':
            photo_exp_stats[group] = {'count': len(data), 'mean_score': data['score'].mean(), 'std_score': data['score'].std(), 'mean_reaction_time': data['reaction_time'].mean(), 'std_reaction_time': data['reaction_time'].std(), 'score_distribution': data['score'].value_counts().sort_index()}
    block_stats = {}
    for block in df['block'].unique():
        if block:
            block_data = df[df['block'] == block]
            block_exp_grouped = block_data.groupby('photo_exp_category')
            block_exp_stats = {}
            for group, data in block_exp_grouped:
                if group != 'Unknown':
                    block_exp_stats[group] = {'count': len(data), 'mean_score': data['score'].mean(), 'std_score': data['score'].std()}
            block_stats[block] = block_exp_stats
    return (photo_exp_stats, block_stats, df)

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

def generate_report(photo_exp_stats, block_stats, anova_result, output_path):
    """Generate analysis report."""
    report = []
    report.append('=' * 80)
    report.append('PHOTOGRAPHY EXPERIENCE EFFECT ON AESTHETIC JUDGMENT ANALYSIS REPORT')
    report.append('=' * 80)
    report.append('')
    report.append('SUMMARY STATISTICS BY PHOTOGRAPHY EXPERIENCE')
    report.append('-' * 80)
    report.append(f'{'Experience Level':<30} {'N':>5} {'Mean Score':>12} {'Std Score':>12} {'Mean RT (ms)':>15}')
    report.append('-' * 80)
    for group, stat in photo_exp_stats.items():
        report.append(f'{group:<30} {stat['count']:>5} {stat['mean_score']:>12.2f} {stat['std_score']:>12.2f} {stat['mean_reaction_time']:>15.1f}')
    report.append('')
    report.append('SUMMARY STATISTICS BY BLOCK TYPE AND PHOTOGRAPHY EXPERIENCE')
    report.append('-' * 80)
    for block, stats in block_stats.items():
        report.append(f'\n{block.capitalize()} Images:')
        report.append(f'{'Experience Level':<30} {'N':>5} {'Mean Score':>12} {'Std Score':>12}')
        report.append('-' * 80)
        for group, stat in stats.items():
            report.append(f'{group:<30} {stat['count']:>5} {stat['mean_score']:>12.2f} {stat['std_score']:>12.2f}')
    report.append('')
    report.append('STATISTICAL TESTS')
    report.append('-' * 80)
    if anova_result[0] is not None:
        report.append(f'Photography Experience ANOVA F-statistic: {anova_result[0]:.4f}')
        report.append(f'Photography Experience ANOVA p-value: {anova_result[1]:.4f}')
    report.append('')
    report.append('KEY FINDINGS')
    report.append('-' * 80)
    exp_mean_scores = {group: stat['mean_score'] for group, stat in photo_exp_stats.items()}
    sorted_exp = sorted(exp_mean_scores.items(), key=lambda x: x[1], reverse=True)
    report.append('\nScore Analysis by Photography Experience:')
    for group, mean_score in sorted_exp:
        report.append(f'  {group}: {mean_score:.2f}')
    exp_mean_rt = {group: stat['mean_reaction_time'] for group, stat in photo_exp_stats.items()}
    sorted_exp_rt = sorted(exp_mean_rt.items(), key=lambda x: x[1])
    report.append('\nReaction Time Analysis by Photography Experience:')
    for group, rt in sorted_exp_rt:
        report.append(f'  {group}: {rt:.1f} ms')
    report.append('\nINTERPRETATION')
    report.append('-' * 80)
    if anova_result[0] is not None and anova_result[1] < 0.05:
        report.append('\nStatistically significant differences found between photography experience groups.')
    else:
        report.append('\nNo statistically significant differences found between photography experience groups.')
    report.append('\nADDITIONAL INSIGHTS')
    report.append('-' * 80)
    report.append('\nKey Question Analysis:')
    report.append('1. Does photography experience affect image aesthetic judgments?')
    if exp_mean_scores:
        scores = list(exp_mean_scores.values())
        if max(scores) - min(scores) > 0.1:
            report.append('   - Yes, photography experience appears to affect aesthetic judgments')
            highest_group = max(exp_mean_scores, key=exp_mean_scores.get)
            lowest_group = min(exp_mean_scores, key=exp_mean_scores.get)
            report.append(f'   - Highest scoring group: {highest_group} ({exp_mean_scores[highest_group]:.2f})')
            report.append(f'   - Lowest scoring group: {lowest_group} ({exp_mean_scores[lowest_group]:.2f})')
        else:
            report.append('   - No significant effect of photography experience on aesthetic judgments')
    report.append('\nPossible Explanations:')
    report.append('  - Photography experience may develop a more discerning eye for visual composition')
    report.append('  - Different experience levels may have different aesthetic standards')
    report.append('  - Technical knowledge of photography may influence how individuals evaluate images')
    report.append('')
    report.append('=' * 80)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    print(f'Analysis report saved: {output_path}')

def main():
    """Main function to run photography experience analysis."""
    print('Reading data...')
    df = read_data()
    print(f'Total records: {len(df)}')
    print('\nAnalyzing photography experience effects...')
    photo_exp_stats, block_stats, df = analyze_photo_exp_effect(df)
    print('\nGenerating visualizations...')
    os.makedirs('code/picture', exist_ok=True)
    print('\nPerforming statistical tests...')
    anova_result = perform_statistical_tests(df, 'photo_exp_category')
    generate_report(photo_exp_stats, block_stats, anova_result, 'code/photo_exp_analysis_report.txt')
    print('\n' + '=' * 80)
    print('PHOTOGRAPHY EXPERIENCE EFFECT ANALYSIS COMPLETE')
    print('=' * 80)
    print('\nGenerated files:')
    print('  - code/picture/photo_exp_analysis.png')
    print('  - code/picture/photo_exp_reaction_time.png')
    print('  - code/picture/photo_exp_score_distribution.png')
    print('  - code/picture/photo_exp_block_analysis.png')
    print('  - code/photo_exp_analysis_report.txt')
if __name__ == '__main__':
    main()
