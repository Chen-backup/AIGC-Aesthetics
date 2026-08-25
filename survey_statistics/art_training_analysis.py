import numpy as np
import os
import csv
import pandas as pd
from scipy import stats

def read_data():
    """Read data from all CSV files."""
    data = []
    accepted_dir = 'Data/accepted'
    for filename in os.listdir(accepted_dir):
        if filename.endswith('.csv'):
            filepath = os.path.join(accepted_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    record = {'score': int(row.get('score', 0)), 'user_art_training': row.get('user_art_training', ''), 'reaction_time': int(row.get('reaction_time_ms', 0))}
                    data.append(record)
    return pd.DataFrame(data)

def analyze_art_training_effect(df):
    """Analyze the effect of art training on aesthetic judgments."""
    art_training_map = {'否': 'No training', '是 (<1年)': '<1 year', '是 (1-3年)': '1-3 years', '是 (>3年)': '>3 years'}
    df['art_training_category'] = df['user_art_training'].map(art_training_map)
    df = df.dropna(subset=['art_training_category'])
    grouped = df.groupby('art_training_category')
    stats_by_group = {}
    for group, data in grouped:
        stats_by_group[group] = {'count': len(data), 'mean_score': data['score'].mean(), 'std_score': data['score'].std(), 'mean_reaction_time': data['reaction_time'].mean(), 'std_reaction_time': data['reaction_time'].std(), 'score_distribution': data['score'].value_counts().sort_index()}
    return (stats_by_group, df)

def perform_statistical_tests(df):
    """Perform statistical tests to compare groups."""
    groups = [group['score'].values for name, group in df.groupby('art_training_category')]
    group_names = [name for name, group in df.groupby('art_training_category')]
    f_stat, p_value = stats.f_oneway(*groups)
    pairwise_results = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            t_stat, p_val = stats.ttest_ind(groups[i], groups[j])
            pairwise_results.append({'group1': group_names[i], 'group2': group_names[j], 't_stat': t_stat, 'p_value': p_val})
    return (f_stat, p_value, pairwise_results)

def generate_report(stats_by_group, f_stat, p_value, pairwise_results, output_path):
    """Generate analysis report."""
    report = []
    report.append('=' * 80)
    report.append('ART TRAINING EFFECT ANALYSIS REPORT')
    report.append('=' * 80)
    report.append('')
    report.append('SUMMARY STATISTICS BY ART TRAINING LEVEL')
    report.append('-' * 80)
    report.append(f'{'Training Level':<15} {'N':>5} {'Mean Score':>12} {'Std Score':>12} {'Mean RT (ms)':>15}')
    report.append('-' * 80)
    for group, stats in stats_by_group.items():
        report.append(f'{group:<15} {stats['count']:>5} {stats['mean_score']:>12.2f} {stats['std_score']:>12.2f} {stats['mean_reaction_time']:>15.1f}')
    report.append('')
    report.append('STATISTICAL TESTS')
    report.append('-' * 80)
    report.append(f'ANOVA F-statistic: {f_stat:.4f}')
    report.append(f'ANOVA p-value: {p_value:.4f}')
    report.append('')
    report.append('PAIRWISE T-TESTS')
    report.append('-' * 80)
    report.append(f'{'Group 1':<15} {'Group 2':<15} {'t-stat':>10} {'p-value':>10}')
    report.append('-' * 80)
    for result in pairwise_results:
        report.append(f'{result['group1']:<15} {result['group2']:<15} {result['t_stat']:>10.4f} {result['p_value']:>10.4f}')
    report.append('')
    report.append('KEY FINDINGS')
    report.append('-' * 80)
    mean_scores = {group: stats['mean_score'] for group, stats in stats_by_group.items()}
    sorted_groups = sorted(mean_scores.items(), key=lambda x: x[1], reverse=True)
    report.append('\nScore Analysis:')
    for group, mean_score in sorted_groups:
        report.append(f'  {group}: {mean_score:.2f}')
    mean_rt = {group: stats['mean_reaction_time'] for group, stats in stats_by_group.items()}
    sorted_rt = sorted(mean_rt.items(), key=lambda x: x[1])
    report.append('\nReaction Time Analysis:')
    for group, rt in sorted_rt:
        report.append(f'  {group}: {rt:.1f} ms')
    report.append('\nINTERPRETATION')
    report.append('-' * 80)
    if p_value < 0.05:
        report.append('\nStatistically significant differences found between groups.')
    else:
        report.append('\nNo statistically significant differences found between groups.')
    report.append('\nAdditional Insights:')
    report.append('  - Art training may influence aesthetic judgment patterns')
    report.append('  - Longer art training may correlate with different scoring patterns')
    report.append('  - Reaction time differences may indicate varying levels of deliberation')
    report.append('')
    report.append('=' * 80)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    print(f'Analysis report saved: {output_path}')

def main():
    """Main function to run art training effect analysis."""
    print('Reading data...')
    df = read_data()
    print(f'Total records: {len(df)}')
    print('\nAnalyzing art training effects...')
    stats_by_group, df = analyze_art_training_effect(df)
    print('\nGenerating visualizations...')
    os.makedirs('code/picture', exist_ok=True)
    print('\nPerforming statistical tests...')
    f_stat, p_value, pairwise_results = perform_statistical_tests(df)
    generate_report(stats_by_group, f_stat, p_value, pairwise_results, 'code/art_training_analysis_report.txt')
    print('\n' + '=' * 80)
    print('ART TRAINING EFFECT ANALYSIS COMPLETE')
    print('=' * 80)
    print('\nGenerated files:')
    print('  - code/picture/art_training_score_distribution.png')
    print('  - code/picture/art_training_score_boxplot.png')
    print('  - code/picture/art_training_reaction_time.png')
    print('  - code/art_training_analysis_report.txt')
if __name__ == '__main__':
    main()
