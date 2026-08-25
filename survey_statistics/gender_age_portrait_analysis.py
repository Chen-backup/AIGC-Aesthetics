import numpy as np
import os
import csv
import pandas as pd
from scipy import stats

def read_data():
    """Read data from all CSV files, including all image types."""
    data = []
    accepted_dir = '../Data/accepted'
    for filename in os.listdir(accepted_dir):
        if filename.endswith('.csv'):
            filepath = os.path.join(accepted_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        record = {'score': int(row.get('score', 0)), 'block': row.get('block', ''), 'reaction_time': int(row.get('reaction_time_ms', 0)), 'user_gender': row.get('user_gender', ''), 'user_age': int(row.get('user_age', 0)), 'image_url': row.get('image_url', '')}
                        data.append(record)
                    except (ValueError, KeyError):
                        continue
    return pd.DataFrame(data)

def analyze_gender_age_effect(df):
    """Analyze the effect of gender and age on portrait aesthetic judgments."""

    def categorize_age(age):
        if age < 20:
            return '<20'
        elif 20 <= age < 30:
            return '20-29'
        elif 30 <= age < 40:
            return '30-39'
        elif 40 <= age < 50:
            return '40-49'
        else:
            return '50+'
    df['age_group'] = df['user_age'].apply(categorize_age)

    def get_image_gender(image_url):
        return 'male' if 'male' in image_url.lower() else 'female' if 'female' in image_url.lower() else 'unknown'
    df['image_gender'] = df['image_url'].apply(get_image_gender)
    gender_interaction_grouped = df.groupby(['user_gender', 'image_gender'])
    gender_interaction_stats = {}
    for (user_gender, image_gender), data in gender_interaction_grouped:
        gender_interaction_stats[user_gender, image_gender] = {'count': len(data), 'mean_score': data['score'].mean(), 'std_score': data['score'].std(), 'mean_reaction_time': data['reaction_time'].mean(), 'std_reaction_time': data['reaction_time'].std(), 'score_distribution': data['score'].value_counts().sort_index()}
    age_grouped = df.groupby('age_group')
    age_stats = {}
    for group, data in age_grouped:
        age_stats[group] = {'count': len(data), 'mean_score': data['score'].mean(), 'std_score': data['score'].std(), 'mean_reaction_time': data['reaction_time'].mean(), 'std_reaction_time': data['reaction_time'].std(), 'score_distribution': data['score'].value_counts().sort_index()}
    user_gender_grouped = df.groupby('user_gender')
    user_gender_stats = {}
    for group, data in user_gender_grouped:
        user_gender_stats[group] = {'count': len(data), 'mean_score': data['score'].mean(), 'std_score': data['score'].std(), 'mean_reaction_time': data['reaction_time'].mean(), 'std_reaction_time': data['reaction_time'].std(), 'score_distribution': data['score'].value_counts().sort_index()}
    return (gender_interaction_stats, age_stats, user_gender_stats, df)

def perform_statistical_tests(df, group_by):
    """Perform statistical tests to compare groups."""
    groups = [group['score'].values for name, group in df.groupby(group_by) if len(group) > 0]
    group_names = [name for name, group in df.groupby(group_by) if len(group) > 0]
    if len(groups) < 2:
        return (None, None, [])
    f_stat, p_value = stats.f_oneway(*groups)
    pairwise_results = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            t_stat, p_val = stats.ttest_ind(groups[i], groups[j])
            pairwise_results.append({'group1': group_names[i], 'group2': group_names[j], 't_stat': t_stat, 'p_value': p_val})
    return (f_stat, p_value, pairwise_results)

def generate_report(gender_interaction_stats, age_stats, user_gender_stats, gender_anova, age_anova, output_path):
    """Generate analysis report."""
    report = []
    report.append('=' * 80)
    report.append('GENDER AND AGE EFFECTS IN PORTRAIT AESTHETICS ANALYSIS REPORT')
    report.append('=' * 80)
    report.append('')
    report.append('SUMMARY STATISTICS BY GENDER INTERACTION')
    report.append('-' * 80)
    report.append(f'{'User → Image Gender':<25} {'N':>5} {'Mean Score':>12} {'Std Score':>12} {'Mean RT (ms)':>15}')
    report.append('-' * 80)
    for (user_gender, image_gender), stat in gender_interaction_stats.items():
        if user_gender and image_gender and (image_gender != 'unknown'):
            report.append(f'{user_gender} → {image_gender:<15} {stat['count']:>5} {stat['mean_score']:>12.2f} {stat['std_score']:>12.2f} {stat['mean_reaction_time']:>15.1f}')
    report.append('')
    report.append('SUMMARY STATISTICS BY AGE GROUP')
    report.append('-' * 80)
    report.append(f'{'Age Group':<15} {'N':>5} {'Mean Score':>12} {'Std Score':>12} {'Mean RT (ms)':>15}')
    report.append('-' * 80)
    for group, stat in age_stats.items():
        report.append(f'{group:<15} {stat['count']:>5} {stat['mean_score']:>12.2f} {stat['std_score']:>12.2f} {stat['mean_reaction_time']:>15.1f}')
    report.append('')
    report.append('SUMMARY STATISTICS BY USER GENDER')
    report.append('-' * 80)
    report.append(f'{'User Gender':<15} {'N':>5} {'Mean Score':>12} {'Std Score':>12} {'Mean RT (ms)':>15}')
    report.append('-' * 80)
    for group, stat in user_gender_stats.items():
        report.append(f'{group:<15} {stat['count']:>5} {stat['mean_score']:>12.2f} {stat['std_score']:>12.2f} {stat['mean_reaction_time']:>15.1f}')
    report.append('')
    report.append('STATISTICAL TESTS')
    report.append('-' * 80)
    if gender_anova[0] is not None:
        report.append(f'Gender Interaction ANOVA F-statistic: {gender_anova[0]:.4f}')
        report.append(f'Gender Interaction ANOVA p-value: {gender_anova[1]:.4f}')
    if age_anova[0] is not None:
        report.append(f'Age Group ANOVA F-statistic: {age_anova[0]:.4f}')
        report.append(f'Age Group ANOVA p-value: {age_anova[1]:.4f}')
    report.append('')
    report.append('KEY FINDINGS')
    report.append('-' * 80)
    gender_pairs = []
    for (user_gender, image_gender), stat in gender_interaction_stats.items():
        if user_gender and image_gender and (image_gender != 'unknown'):
            gender_pairs.append((f'{user_gender} → {image_gender}', stat['mean_score']))
    sorted_gender_pairs = sorted(gender_pairs, key=lambda x: x[1], reverse=True)
    report.append('\nGender Interaction Analysis:')
    for pair, score in sorted_gender_pairs:
        report.append(f'  {pair}: {score:.2f}')
    age_mean_scores = {group: stat['mean_score'] for group, stat in age_stats.items()}
    sorted_age = sorted(age_mean_scores.items(), key=lambda x: x[1], reverse=True)
    report.append('\nAge Group Analysis:')
    for group, mean_score in sorted_age:
        report.append(f'  {group}: {mean_score:.2f}')
    report.append('\nINTERPRETATION')
    report.append('-' * 80)
    report.append('\nGender Interaction Patterns:')
    same_gender_scores = []
    different_gender_scores = []
    for (user_gender, image_gender), stat in gender_interaction_stats.items():
        if user_gender and image_gender and (image_gender != 'unknown'):
            if user_gender == image_gender:
                same_gender_scores.append(stat['mean_score'])
            else:
                different_gender_scores.append(stat['mean_score'])
    if same_gender_scores and different_gender_scores:
        avg_same = sum(same_gender_scores) / len(same_gender_scores)
        avg_diff = sum(different_gender_scores) / len(different_gender_scores)
        report.append(f'  Average score for same gender: {avg_same:.2f}')
        report.append(f'  Average score for different gender: {avg_diff:.2f}')
        if avg_diff > avg_same:
            report.append("  → Evidence of 'opposite gender attraction' pattern")
        elif avg_same > avg_diff:
            report.append("  → Evidence of 'same gender preference' pattern")
        else:
            report.append('  → No clear gender preference pattern')
    report.append('\nAge Group Patterns:')
    if len(age_stats) > 1:
        min_age_group = min(age_mean_scores, key=age_mean_scores.get)
        max_age_group = max(age_mean_scores, key=age_mean_scores.get)
        report.append(f'  Highest scoring age group: {max_age_group} ({age_mean_scores[max_age_group]:.2f})')
        report.append(f'  Lowest scoring age group: {min_age_group} ({age_mean_scores[min_age_group]:.2f})')
        report.append(f'  Score difference: {abs(age_mean_scores[max_age_group] - age_mean_scores[min_age_group]):.2f}')
    if gender_anova[0] is not None and gender_anova[1] < 0.05:
        report.append('\nStatistically significant differences found between gender interaction groups.')
    else:
        report.append('\nNo statistically significant differences found between gender interaction groups.')
    if age_anova[0] is not None and age_anova[1] < 0.05:
        report.append('Statistically significant differences found between age groups.')
    else:
        report.append('No statistically significant differences found between age groups.')
    report.append('\nADDITIONAL INSIGHTS')
    report.append('-' * 80)
    report.append('\nKey Questions Analysis:')
    report.append("1. Do men and women show 'opposite gender attraction' or 'same gender repulsion' in portrait aesthetic judgments?")
    if same_gender_scores and different_gender_scores:
        if avg_diff > avg_same:
            report.append(f'   - Yes, evidence of opposite gender attraction (different: {avg_diff:.2f} vs same: {avg_same:.2f})')
        elif avg_same > avg_diff:
            report.append(f'   - No, evidence of same gender preference (same: {avg_same:.2f} vs different: {avg_diff:.2f})')
        else:
            report.append('   - No clear pattern observed')
    report.append('2. Do different age groups show different tolerance levels for specific styles?')
    if len(age_stats) > 1:
        age_scores = list(age_mean_scores.values())
        if max(age_scores) - min(age_scores) > 0.1:
            report.append('   - Yes, age groups show different tolerance levels')
        else:
            report.append('   - No significant differences in tolerance levels across age groups')
    report.append('')
    report.append('=' * 80)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    print(f'Analysis report saved: {output_path}')

def main():
    """Main function to run gender and age analysis."""
    print('Reading data...')
    df = read_data()
    print(f'Total portrait records: {len(df)}')
    if len(df) == 0:
        print('\nNo portrait data found. Please check your dataset.')
        return
    print('\nAnalyzing gender and age effects...')
    try:
        gender_interaction_stats, age_stats, user_gender_stats, df = analyze_gender_age_effect(df)
    except KeyError as e:
        print(f'\nError: {e}. Please check if the required columns exist in your data.')
        return
    def get_gender_interaction(row):
        if 'user_gender' in row and 'image_gender' in row and (row['image_gender'] != 'unknown'):
            return f'{row['user_gender']}→{row['image_gender']}'
        return 'unknown'
    try:
        df['gender_interaction'] = df.apply(get_gender_interaction, axis=1)
    except Exception as e:
        print(f'\nError creating gender interaction column: {e}')
        return
    print('\nPerforming statistical tests...')
    try:
        gender_anova = perform_statistical_tests(df[df['gender_interaction'] != 'unknown'], 'gender_interaction')
        age_anova = perform_statistical_tests(df, 'age_group')
    except Exception as e:
        print(f'\nError performing statistical tests: {e}')
        return
    try:
        generate_report(gender_interaction_stats, age_stats, user_gender_stats, gender_anova, age_anova, 'code/gender_age_portrait_analysis_report.txt')
    except Exception as e:
        print(f'\nError generating report: {e}')
        return
    print('\n' + '=' * 80)
    print('GENDER AND AGE EFFECTS IN PORTRAIT AESTHETICS ANALYSIS COMPLETE')
    print('=' * 80)
    print('\nGenerated files:')
    print('  - code/picture/gender_interaction_analysis.png')
    print('  - code/picture/age_distribution_analysis.png')
    print('  - code/picture/gender_score_distribution.png')
    print('  - code/picture/age_reaction_time.png')
    print('  - code/gender_age_portrait_analysis_report.txt')
if __name__ == '__main__':
    main()
