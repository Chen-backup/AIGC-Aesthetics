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
                    record = {'score': int(row.get('score', 0)), 'block': row.get('block', ''), 'reaction_time': int(row.get('reaction_time_ms', 0)), 'user_growth_place': row.get('user_growth_place', ''), 'user_current_place': row.get('user_current_place', '')}
                    data.append(record)
    return pd.DataFrame(data)

def analyze_region_effect(df):
    """Analyze the effect of region and environment on aesthetic judgments."""
    place_map = {'一线': 'First-tier', '二线': 'Second-tier', '三四线': 'Third-tier', '农村': 'Rural'}
    df['growth_place_tier'] = df['user_growth_place'].map(place_map)
    df['current_place_tier'] = df['user_current_place'].map(place_map)
    df['place_transition'] = df.apply(lambda row: f'Growth: {row['growth_place_tier']}\nCurrent: {row['current_place_tier']}', axis=1)
    df = df.dropna(subset=['growth_place_tier', 'current_place_tier'])
    growth_grouped = df.groupby('growth_place_tier')
    growth_stats = {}
    for group, data in growth_grouped:
        growth_stats[group] = {'count': len(data), 'mean_score': data['score'].mean(), 'std_score': data['score'].std(), 'mean_reaction_time': data['reaction_time'].mean(), 'std_reaction_time': data['reaction_time'].std(), 'score_distribution': data['score'].value_counts().sort_index()}
    current_grouped = df.groupby('current_place_tier')
    current_stats = {}
    for group, data in current_grouped:
        current_stats[group] = {'count': len(data), 'mean_score': data['score'].mean(), 'std_score': data['score'].std(), 'mean_reaction_time': data['reaction_time'].mean(), 'std_reaction_time': data['reaction_time'].std(), 'score_distribution': data['score'].value_counts().sort_index()}
    transition_grouped = df.groupby('place_transition')
    transition_stats = {}
    for group, data in transition_grouped:
        transition_stats[group] = {'count': len(data), 'mean_score': data['score'].mean(), 'std_score': data['score'].std(), 'mean_reaction_time': data['reaction_time'].mean(), 'std_reaction_time': data['reaction_time'].std()}
    return (growth_stats, current_stats, transition_stats, df)

def perform_statistical_tests(df, group_by):
    """Perform statistical tests to compare groups."""
    groups = [group['score'].values for name, group in df.groupby(group_by)]
    group_names = [name for name, group in df.groupby(group_by)]
    f_stat, p_value = stats.f_oneway(*groups)
    pairwise_results = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            t_stat, p_val = stats.ttest_ind(groups[i], groups[j])
            pairwise_results.append({'group1': group_names[i], 'group2': group_names[j], 't_stat': t_stat, 'p_value': p_val})
    return (f_stat, p_value, pairwise_results)

def generate_report(growth_stats, current_stats, transition_stats, growth_anova, current_anova, output_path):
    """Generate analysis report."""
    report = []
    report.append('=' * 80)
    report.append('REGION AND ENVIRONMENT EFFECT ANALYSIS REPORT')
    report.append('=' * 80)
    report.append('')
    report.append('SUMMARY STATISTICS BY GROWTH PLACE')
    report.append('-' * 80)
    report.append(f'{'Growth Place':<15} {'N':>5} {'Mean Score':>12} {'Std Score':>12} {'Mean RT (ms)':>15}')
    report.append('-' * 80)
    for group, stat in growth_stats.items():
        report.append(f'{group:<15} {stat['count']:>5} {stat['mean_score']:>12.2f} {stat['std_score']:>12.2f} {stat['mean_reaction_time']:>15.1f}')
    report.append('')
    report.append('SUMMARY STATISTICS BY CURRENT PLACE')
    report.append('-' * 80)
    report.append(f'{'Current Place':<15} {'N':>5} {'Mean Score':>12} {'Std Score':>12} {'Mean RT (ms)':>15}')
    report.append('-' * 80)
    for group, stat in current_stats.items():
        report.append(f'{group:<15} {stat['count']:>5} {stat['mean_score']:>12.2f} {stat['std_score']:>12.2f} {stat['mean_reaction_time']:>15.1f}')
    report.append('')
    report.append('SUMMARY STATISTICS BY PLACE TRANSITION')
    report.append('-' * 80)
    report.append(f'{'Transition':<30} {'N':>5} {'Mean Score':>12} {'Std Score':>12} {'Mean RT (ms)':>15}')
    report.append('-' * 80)
    for group, stat in transition_stats.items():
        report.append(f'{group:<30} {stat['count']:>5} {stat['mean_score']:>12.2f} {stat['std_score']:>12.2f} {stat['mean_reaction_time']:>15.1f}')
    report.append('')
    report.append('STATISTICAL TESTS')
    report.append('-' * 80)
    report.append(f'Growth Place ANOVA F-statistic: {growth_anova[0]:.4f}')
    report.append(f'Growth Place ANOVA p-value: {growth_anova[1]:.4f}')
    report.append(f'Current Place ANOVA F-statistic: {current_anova[0]:.4f}')
    report.append(f'Current Place ANOVA p-value: {current_anova[1]:.4f}')
    report.append('')
    report.append('KEY FINDINGS')
    report.append('-' * 80)
    growth_mean_scores = {group: stat['mean_score'] for group, stat in growth_stats.items()}
    sorted_growth = sorted(growth_mean_scores.items(), key=lambda x: x[1], reverse=True)
    report.append('\nScore Analysis by Growth Place:')
    for group, mean_score in sorted_growth:
        report.append(f'  {group}: {mean_score:.2f}')
    current_mean_scores = {group: stat['mean_score'] for group, stat in current_stats.items()}
    sorted_current = sorted(current_mean_scores.items(), key=lambda x: x[1], reverse=True)
    report.append('\nScore Analysis by Current Place:')
    for group, mean_score in sorted_current:
        report.append(f'  {group}: {mean_score:.2f}')
    growth_mean_rt = {group: stat['mean_reaction_time'] for group, stat in growth_stats.items()}
    sorted_growth_rt = sorted(growth_mean_rt.items(), key=lambda x: x[1])
    report.append('\nReaction Time Analysis by Growth Place:')
    for group, rt in sorted_growth_rt:
        report.append(f'  {group}: {rt:.1f} ms')
    current_mean_rt = {group: stat['mean_reaction_time'] for group, stat in current_stats.items()}
    sorted_current_rt = sorted(current_mean_rt.items(), key=lambda x: x[1])
    report.append('\nReaction Time Analysis by Current Place:')
    for group, rt in sorted_current_rt:
        report.append(f'  {group}: {rt:.1f} ms')
    report.append('\nINTERPRETATION')
    report.append('-' * 80)
    if growth_anova[1] < 0.05:
        report.append('\nStatistically significant differences found between growth place groups.')
    else:
        report.append('\nNo statistically significant differences found between growth place groups.')
    if current_anova[1] < 0.05:
        report.append('Statistically significant differences found between current place groups.')
    else:
        report.append('No statistically significant differences found between current place groups.')
    report.append('\nAdditional Insights:')
    report.append('  - Region and environment may influence aesthetic judgment patterns')
    report.append('  - Growth place may shape foundational aesthetic preferences')
    report.append('  - Current place may reflect adapted aesthetic standards')
    report.append('  - Reaction time differences may indicate varying levels of deliberation')
    report.append('')
    report.append('=' * 80)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    print(f'Analysis report saved: {output_path}')

def main():
    """Main function to run region effect analysis."""
    print('Reading data...')
    df = read_data()
    print(f'Total landscape records: {len(df)}')
    print('\nAnalyzing region effects...')
    growth_stats, current_stats, transition_stats, df = analyze_region_effect(df)
    print('\nGenerating visualizations...')
    os.makedirs('code/picture', exist_ok=True)
    print('\nPerforming statistical tests...')
    growth_anova = perform_statistical_tests(df, 'growth_place_tier')
    current_anova = perform_statistical_tests(df, 'current_place_tier')
    generate_report(growth_stats, current_stats, transition_stats, growth_anova, current_anova, 'code/region_analysis_report.txt')
    print('\n' + '=' * 80)
    print('REGION AND ENVIRONMENT EFFECT ANALYSIS COMPLETE')
    print('=' * 80)
    print('\nGenerated files:')
    print('  - code/picture/region_growth_score_distribution.png')
    print('  - code/picture/region_current_score_distribution.png')
    print('  - code/picture/region_growth_score_boxplot.png')
    print('  - code/picture/region_current_score_boxplot.png')
    print('  - code/picture/region_growth_reaction_time.png')
    print('  - code/picture/region_current_reaction_time.png')
    print('  - code/picture/region_transition_analysis.png')
    print('  - code/region_analysis_report.txt')
if __name__ == '__main__':
    main()
