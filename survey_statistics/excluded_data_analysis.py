"""
Excluded Data Analysis
This script analyzes the excluded data to evaluate the合理性 of manual exclusion.
"""
import os
import csv
import numpy as np
from collections import Counter, defaultdict

def analyze_excluded_files():
    """Analyze all excluded files."""
    excluded_dir = 'Data/excluded'
    files = [f for f in os.listdir(excluded_dir) if f.endswith('.csv')]
    print(f'Total excluded files: {len(files)}')
    total_records = 0
    user_analysis = []
    check_bias_counter = Counter()
    check_ai_ratio_counter = Counter()
    for filename in files:
        filepath = os.path.join(excluded_dir, filename)
        user_data = analyze_single_file(filepath)
        if user_data:
            user_analysis.append(user_data)
            total_records += user_data['n_records']
            check_bias_counter[user_data.get('check_bias', 'N/A')] += 1
            check_ai_ratio_counter[user_data.get('check_ai_ratio', 'N/A')] += 1
    return {'total_files': len(files), 'total_records': total_records, 'user_analysis': user_analysis, 'check_bias_distribution': check_bias_counter, 'check_ai_ratio_distribution': check_ai_ratio_counter}

def analyze_single_file(filepath):
    """Analyze a single excluded file."""
    records = []
    user_info = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i == 0:
                user_info = {'gender': row.get('user_gender', ''), 'age': row.get('user_age', ''), 'education': row.get('user_education', ''), 'art_training': row.get('user_art_training', ''), 'check_bias': row.get('check_bias', ''), 'check_ai_ratio': row.get('check_ai_ratio', '')}
            if 'score' in row and row['score']:
                try:
                    score = int(row['score'])
                    reaction_time = int(row.get('reaction_time_ms', 0))
                    n_changes = int(row.get('n_changes', 0))
                    records.append({'score': score, 'reaction_time': reaction_time, 'n_changes': n_changes})
                except ValueError:
                    pass
    if not records:
        return None
    scores = [r['score'] for r in records]
    reaction_times = [r['reaction_time'] for r in records]
    n_changes_list = [r['n_changes'] for r in records]
    score_counter = Counter(scores)
    most_common_score, most_common_count = score_counter.most_common(1)[0]
    return {'filename': os.path.basename(filepath), 'n_records': len(records), 'mean_score': np.mean(scores), 'std_score': np.std(scores), 'min_score': min(scores), 'max_score': max(scores), 'most_common_score': most_common_score, 'most_common_percentage': most_common_count / len(records) * 100, 'mean_reaction_time': np.mean(reaction_times), 'std_reaction_time': np.std(reaction_times), 'mean_n_changes': np.mean(n_changes_list), 'all_scores_same': len(set(scores)) == 1, **user_info}

def generate_report(analysis):
    """Generate analysis report."""
    report = []
    report.append('=' * 80)
    report.append('EXCLUDED DATA ANALYSIS REPORT')
    report.append('=' * 80)
    report.append('')
    report.append(f'Total excluded files: {analysis['total_files']}')
    report.append(f'Total excluded records: {analysis['total_records']}')
    report.append(f'Average records per file: {analysis['total_records'] / analysis['total_files']:.1f}')
    report.append('')
    report.append('-' * 80)
    report.append('CHECK BIAS DISTRIBUTION')
    report.append('-' * 80)
    for bias, count in analysis['check_bias_distribution'].items():
        percentage = count / analysis['total_files'] * 100
        report.append(f'  {bias}: {count} ({percentage:.1f}%)')
    report.append('')
    report.append('-' * 80)
    report.append('CHECK AI RATIO DISTRIBUTION')
    report.append('-' * 80)
    for ratio, count in analysis['check_ai_ratio_distribution'].items():
        percentage = count / analysis['total_files'] * 100
        report.append(f'  {ratio}: {count} ({percentage:.1f}%)')
    report.append('')
    report.append('-' * 80)
    report.append('USER ANALYSIS')
    report.append('-' * 80)
    if analysis['user_analysis']:
        all_scores_same = sum((1 for u in analysis['user_analysis'] if u['all_scores_same']))
        high_ai_ratio = sum((1 for u in analysis['user_analysis'] if u.get('check_ai_ratio', '') == '50%'))
        bias_yes = sum((1 for u in analysis['user_analysis'] if u.get('check_bias', '') == '是'))
        report.append(f'Users with all scores the same: {all_scores_same} ({all_scores_same / len(analysis['user_analysis']) * 100:.1f}%)')
        report.append(f'Users with high AI ratio (50%): {high_ai_ratio} ({high_ai_ratio / len(analysis['user_analysis']) * 100:.1f}%)')
        report.append(f'Users with bias flag: {bias_yes} ({bias_yes / len(analysis['user_analysis']) * 100:.1f}%)')
        report.append('')
        sorted_users = sorted(analysis['user_analysis'], key=lambda x: x['most_common_percentage'], reverse=True)[:10]
        report.append('Top 10 users with highest score consistency:')
        for i, user in enumerate(sorted_users, 1):
            report.append(f'  {i}. {user['filename']}: {user['most_common_score']} ({user['most_common_percentage']:.1f}%)')
        report.append('')
    report.append('-' * 80)
    report.append('EVALUATION OF EXCLUSION REASONING')
    report.append('-' * 80)
    if analysis['user_analysis']:
        all_scores_same_ratio = sum((1 for u in analysis['user_analysis'] if u['all_scores_same'])) / len(analysis['user_analysis']) * 100
        high_ai_ratio = sum((1 for u in analysis['user_analysis'] if u.get('check_ai_ratio', '') == '50%')) / len(analysis['user_analysis']) * 100
        bias_yes_ratio = sum((1 for u in analysis['user_analysis'] if u.get('check_bias', '') == '是')) / len(analysis['user_analysis']) * 100
        report.append('Based on analysis, the exclusion appears to be based on:')
        if all_scores_same_ratio > 20:
            report.append('  ✓ Score pattern异常 (e.g., all scores the same)')
        if high_ai_ratio > 20:
            report.append('  ✓ High AI ratio (50%)')
        if bias_yes_ratio > 20:
            report.append('  ✓ Bias detection')
        valid_exclusion = 0
        if all_scores_same_ratio > 30:
            valid_exclusion += 1
        if high_ai_ratio > 30:
            valid_exclusion += 1
        if bias_yes_ratio > 30:
            valid_exclusion += 1
        if valid_exclusion >= 2:
            report.append('\n✓ Exclusion appears to be reasonable based on multiple criteria')
        elif valid_exclusion == 1:
            report.append('\n⚠ Exclusion appears to be based on limited criteria')
        else:
            report.append('\n✗ Exclusion criteria not clearly identified')
    report.append('')
    report.append('=' * 80)
    return report

def main():
    """Main function."""
    print('Analyzing excluded data...')
    analysis = analyze_excluded_files()
    print('\nGenerating report...')
    report = generate_report(analysis)
    output_path = 'code/excluded_data_analysis.txt'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    print('\n' + '=' * 80)
    print('EXCLUDED DATA ANALYSIS COMPLETE')
    print('=' * 80)
    print(f'\nReport saved to: {output_path}')
    print('\nSUMMARY:')
    print(f'Total excluded files: {analysis['total_files']}')
    print(f'Total excluded records: {analysis['total_records']}')
    print(f'Check bias distribution: {dict(analysis['check_bias_distribution'])}')
    print(f'Check AI ratio distribution: {dict(analysis['check_ai_ratio_distribution'])}')
if __name__ == '__main__':
    main()
