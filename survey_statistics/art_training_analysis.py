import matplotlib.pyplot as plt
import numpy as np
import os
import csv
import pandas as pd
from scipy import stats

# Set font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

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
                    # Extract relevant features
                    record = {
                        'score': int(row.get('score', 0)),
                        'user_art_training': row.get('user_art_training', ''),
                        'reaction_time': int(row.get('reaction_time_ms', 0)),
                    }
                    data.append(record)
    
    return pd.DataFrame(data)

def analyze_art_training_effect(df):
    """Analyze the effect of art training on aesthetic judgments."""
    # Map art training categories
    art_training_map = {
        '否': 'No training',
        '是 (<1年)': '<1 year',
        '是 (1-3年)': '1-3 years',
        '是 (>3年)': '>3 years'
    }
    
    # Create a new column with mapped categories
    df['art_training_category'] = df['user_art_training'].map(art_training_map)
    
    # Remove rows with missing art training data
    df = df.dropna(subset=['art_training_category'])
    
    # Group by art training category
    grouped = df.groupby('art_training_category')
    
    # Calculate statistics for each group
    stats_by_group = {}
    for group, data in grouped:
        stats_by_group[group] = {
            'count': len(data),
            'mean_score': data['score'].mean(),
            'std_score': data['score'].std(),
            'mean_reaction_time': data['reaction_time'].mean(),
            'std_reaction_time': data['reaction_time'].std(),
            'score_distribution': data['score'].value_counts().sort_index(),
        }
    
    return stats_by_group, df

def plot_score_distribution(stats_by_group, output_path):
    """Plot score distribution by art training category."""
    plt.figure(figsize=(12, 8))
    
    # Define colors for each category
    colors = {'No training': '#9EB5FF', '<1 year': '#FF9E9E', '1-3 years': '#87CEEB', '>3 years': '#FFB6C1'}
    
    # Plot score distribution for each group
    for group, stats in stats_by_group.items():
        score_dist = stats['score_distribution']
        plt.bar(score_dist.index - 0.3 + list(stats_by_group.keys()).index(group) * 0.2, 
                score_dist.values, width=0.15, label=group, color=colors[group])
    
    plt.xlabel('Score', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title('Score Distribution by Art Training Level', fontsize=16, fontweight='bold')
    plt.xticks(range(1, 8))
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Score distribution plot saved: {output_path}')

def plot_score_boxplot(df, output_path):
    """Plot boxplot of scores by art training category."""
    plt.figure(figsize=(10, 6))
    
    # Define colors for each category
    colors = {'No training': '#9EB5FF', '<1 year': '#FF9E9E', '1-3 years': '#87CEEB', '>3 years': '#FFB6C1'}
    
    # Create boxplot
    boxplot_data = [group['score'].values for name, group in df.groupby('art_training_category')]
    boxplot_labels = [name for name, group in df.groupby('art_training_category')]
    
    box = plt.boxplot(boxplot_data, labels=boxplot_labels, patch_artist=True)
    
    # Set colors
    for patch, color in zip(box['boxes'], [colors[label] for label in boxplot_labels]):
        patch.set_facecolor(color)
    
    plt.xlabel('Art Training Level', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.title('Score Distribution by Art Training Level', fontsize=16, fontweight='bold')
    plt.ylim(0, 8)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Score boxplot saved: {output_path}')

def plot_reaction_time_comparison(stats_by_group, output_path):
    """Plot reaction time comparison by art training category."""
    plt.figure(figsize=(10, 6))
    
    # Define colors for each category
    colors = {'No training': '#9EB5FF', '<1 year': '#FF9E9E', '1-3 years': '#87CEEB', '>3 years': '#FFB6C1'}
    
    # Prepare data
    groups = list(stats_by_group.keys())
    mean_rt = [stats['mean_reaction_time'] for stats in stats_by_group.values()]
    std_rt = [stats['std_reaction_time'] for stats in stats_by_group.values()]
    
    # Plot
    x_pos = np.arange(len(groups))
    plt.bar(x_pos, mean_rt, yerr=std_rt, capsize=5, color=[colors[group] for group in groups])
    plt.xticks(x_pos, groups)
    plt.xlabel('Art Training Level', fontsize=14)
    plt.ylabel('Mean Reaction Time (ms)', fontsize=14)
    plt.title('Reaction Time by Art Training Level', fontsize=16, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Reaction time comparison plot saved: {output_path}')

def perform_statistical_tests(df):
    """Perform statistical tests to compare groups."""
    # Get groups
    groups = [group['score'].values for name, group in df.groupby('art_training_category')]
    group_names = [name for name, group in df.groupby('art_training_category')]
    
    # Perform ANOVA
    f_stat, p_value = stats.f_oneway(*groups)
    
    # Perform pairwise t-tests
    pairwise_results = []
    for i in range(len(groups)):
        for j in range(i+1, len(groups)):
            t_stat, p_val = stats.ttest_ind(groups[i], groups[j])
            pairwise_results.append({
                'group1': group_names[i],
                'group2': group_names[j],
                't_stat': t_stat,
                'p_value': p_val
            })
    
    return f_stat, p_value, pairwise_results

def generate_report(stats_by_group, f_stat, p_value, pairwise_results, output_path):
    """Generate analysis report."""
    report = []
    report.append("=" * 80)
    report.append("ART TRAINING EFFECT ANALYSIS REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Summary statistics
    report.append("SUMMARY STATISTICS BY ART TRAINING LEVEL")
    report.append("-" * 80)
    report.append(f"{'Training Level':<15} {'N':>5} {'Mean Score':>12} {'Std Score':>12} {'Mean RT (ms)':>15}")
    report.append("-" * 80)
    
    for group, stats in stats_by_group.items():
        report.append(f"{group:<15} {stats['count']:>5} {stats['mean_score']:>12.2f} {stats['std_score']:>12.2f} {stats['mean_reaction_time']:>15.1f}")
    
    report.append("")
    
    # Statistical tests
    report.append("STATISTICAL TESTS")
    report.append("-" * 80)
    report.append(f"ANOVA F-statistic: {f_stat:.4f}")
    report.append(f"ANOVA p-value: {p_value:.4f}")
    report.append("")
    
    report.append("PAIRWISE T-TESTS")
    report.append("-" * 80)
    report.append(f"{'Group 1':<15} {'Group 2':<15} {'t-stat':>10} {'p-value':>10}")
    report.append("-" * 80)
    
    for result in pairwise_results:
        report.append(f"{result['group1']:<15} {result['group2']:<15} {result['t_stat']:>10.4f} {result['p_value']:>10.4f}")
    
    report.append("")
    
    # Key findings
    report.append("KEY FINDINGS")
    report.append("-" * 80)
    
    # Analyze mean scores
    mean_scores = {group: stats['mean_score'] for group, stats in stats_by_group.items()}
    sorted_groups = sorted(mean_scores.items(), key=lambda x: x[1], reverse=True)
    
    report.append("\nScore Analysis:")
    for group, mean_score in sorted_groups:
        report.append(f"  {group}: {mean_score:.2f}")
    
    # Analyze reaction times
    mean_rt = {group: stats['mean_reaction_time'] for group, stats in stats_by_group.items()}
    sorted_rt = sorted(mean_rt.items(), key=lambda x: x[1])
    
    report.append("\nReaction Time Analysis:")
    for group, rt in sorted_rt:
        report.append(f"  {group}: {rt:.1f} ms")
    
    # Interpretation
    report.append("\nINTERPRETATION")
    report.append("-" * 80)
    
    if p_value < 0.05:
        report.append("\nStatistically significant differences found between groups.")
    else:
        report.append("\nNo statistically significant differences found between groups.")
    
    # Additional insights
    report.append("\nAdditional Insights:")
    report.append("  - Art training may influence aesthetic judgment patterns")
    report.append("  - Longer art training may correlate with different scoring patterns")
    report.append("  - Reaction time differences may indicate varying levels of deliberation")
    
    report.append("")
    report.append("=" * 80)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f'Analysis report saved: {output_path}')

def main():
    """Main function to run art training effect analysis."""
    print("Reading data...")
    df = read_data()
    print(f"Total records: {len(df)}")
    
    print("\nAnalyzing art training effects...")
    stats_by_group, df = analyze_art_training_effect(df)
    
    print("\nGenerating visualizations...")
    os.makedirs('code/picture', exist_ok=True)
    
    # Generate plots
    plot_score_distribution(stats_by_group, 'code/picture/art_training_score_distribution.png')
    plot_score_boxplot(df, 'code/picture/art_training_score_boxplot.png')
    plot_reaction_time_comparison(stats_by_group, 'code/picture/art_training_reaction_time.png')
    
    # Perform statistical tests
    print("\nPerforming statistical tests...")
    f_stat, p_value, pairwise_results = perform_statistical_tests(df)
    
    # Generate report
    generate_report(stats_by_group, f_stat, p_value, pairwise_results, 'code/art_training_analysis_report.txt')
    
    print("\n" + "=" * 80)
    print("ART TRAINING EFFECT ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - code/picture/art_training_score_distribution.png")
    print("  - code/picture/art_training_score_boxplot.png")
    print("  - code/picture/art_training_reaction_time.png")
    print("  - code/art_training_analysis_report.txt")

if __name__ == '__main__':
    main()
