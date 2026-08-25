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
    accepted_dir = '../Data/accepted'
    
    for filename in os.listdir(accepted_dir):
        if filename.endswith('.csv'):
            filepath = os.path.join(accepted_dir, filename)
            
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Extract relevant features
                    try:
                        record = {
                            'score': int(row.get('score', 0)),
                            'block': row.get('block', ''),
                            'reaction_time': int(row.get('reaction_time_ms', 0)),
                            'user_consumption': row.get('user_consumption', ''),
                        }
                        data.append(record)
                    except (ValueError, KeyError):
                        # Skip records with invalid data
                        continue
    
    return pd.DataFrame(data)

def analyze_consumption_effect(df):
    """Analyze the effect of monthly consumption on aesthetic judgments."""
    # Process consumption levels
    def categorize_consumption(consumption):
        if not consumption:
            return 'Unknown'
        # Map consumption categories to numerical ranges
        consumption_map = {
            '<1000': 'Low (<1000)',
            '1000-3000': 'Medium (1000-3000)',
            '3000-5000': 'Medium-High (3000-5000)',
            '3000-6000': 'Medium-High (3000-6000)',
            '5000-8000': 'High (5000-8000)',
            '6000-10000': 'High (6000-10000)',
            '>8000': 'Very High (>8000)',
            '10000以上': 'Very High (>10000)'
        }
        return consumption_map.get(consumption, 'Unknown')
    
    df['consumption_category'] = df['user_consumption'].apply(categorize_consumption)
    
    # Group by consumption category
    consumption_grouped = df.groupby('consumption_category')
    consumption_stats = {}
    for group, data in consumption_grouped:
        if group != 'Unknown':
            consumption_stats[group] = {
                'count': len(data),
                'mean_score': data['score'].mean(),
                'std_score': data['score'].std(),
                'mean_reaction_time': data['reaction_time'].mean(),
                'std_reaction_time': data['reaction_time'].std(),
                'score_distribution': data['score'].value_counts().sort_index(),
            }
    
    # Group by block type (image category)
    block_stats = {}
    for block in df['block'].unique():
        if block:
            block_data = df[df['block'] == block]
            block_consumption_grouped = block_data.groupby('consumption_category')
            block_consumption_stats = {}
            for group, data in block_consumption_grouped:
                if group != 'Unknown':
                    block_consumption_stats[group] = {
                        'count': len(data),
                        'mean_score': data['score'].mean(),
                        'std_score': data['score'].std(),
                    }
            block_stats[block] = block_consumption_stats
    
    return consumption_stats, block_stats, df

def plot_consumption_analysis(stats, output_path):
    """Plot consumption analysis."""
    plt.figure(figsize=(12, 8))
    
    # Define colors
    colors = ['#9EB5FF', '#FF9E9E', '#87CEEB', '#FFB6C1', '#90EE90']
    
    # Prepare data
    categories = list(stats.keys())
    mean_scores = [stat['mean_score'] for stat in stats.values()]
    counts = [stat['count'] for stat in stats.values()]
    
    # Plot
    x_pos = np.arange(len(categories))
    bars = plt.bar(x_pos, mean_scores, color=colors[:len(categories)])
    
    # Add count labels on top of bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'n={count}', ha='center', va='bottom')
    
    plt.xticks(x_pos, categories, rotation=45, ha='right')
    plt.xlabel('Monthly Consumption Category', fontsize=14)
    plt.ylabel('Mean Score', fontsize=14)
    plt.title('Effect of Monthly Consumption on Aesthetic Judgment', fontsize=16, fontweight='bold')
    plt.ylim(0, 5)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Consumption analysis plot saved: {output_path}')

def plot_reaction_time_comparison(stats, output_path):
    """Plot reaction time comparison by consumption category."""
    plt.figure(figsize=(10, 6))
    
    # Define colors
    colors = ['#9EB5FF', '#FF9E9E', '#87CEEB', '#FFB6C1', '#90EE90']
    
    # Prepare data
    groups = list(stats.keys())
    mean_rt = [stat['mean_reaction_time'] for stat in stats.values()]
    std_rt = [stat['std_reaction_time'] for stat in stats.values()]
    
    # Plot
    x_pos = np.arange(len(groups))
    plt.bar(x_pos, mean_rt, yerr=std_rt, capsize=5, color=[colors[i % len(colors)] for i in range(len(groups))])
    plt.xticks(x_pos, groups, rotation=45, ha='right')
    plt.xlabel('Monthly Consumption Category', fontsize=14)
    plt.ylabel('Mean Reaction Time (ms)', fontsize=14)
    plt.title('Reaction Time by Monthly Consumption Category', fontsize=16, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Reaction time comparison plot saved: {output_path}')

def plot_score_distribution(stats, title, output_path):
    """Plot score distribution by consumption category."""
    plt.figure(figsize=(12, 8))
    
    # Define colors
    colors = ['#9EB5FF', '#FF9E9E', '#87CEEB', '#FFB6C1', '#90EE90']
    
    # Plot score distribution for each group
    for i, (group, stat) in enumerate(stats.items()):
        score_dist = stat['score_distribution']
        plt.bar(score_dist.index - 0.3 + i * 0.2, 
                score_dist.values, width=0.15, label=group, color=colors[i % len(colors)])
    
    plt.xlabel('Score', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xticks(range(1, 8))
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Score distribution plot saved: {output_path}')

def plot_block_consumption_analysis(block_stats, output_path):
    """Plot consumption analysis by block type."""
    plt.figure(figsize=(14, 10))
    
    # Define colors
    colors = ['#9EB5FF', '#FF9E9E', '#87CEEB', '#FFB6C1', '#90EE90']
    
    # Get all unique consumption categories
    all_categories = set()
    for block, stats in block_stats.items():
        all_categories.update(stats.keys())
    all_categories = sorted(all_categories)
    
    # Plot each block
    blocks = list(block_stats.keys())
    n_blocks = len(blocks)
    
    for i, block in enumerate(blocks):
        plt.subplot(n_blocks, 1, i+1)
        stats = block_stats[block]
        
        # Prepare data
        categories = []
        scores = []
        counts = []
        
        for cat in all_categories:
            if cat in stats:
                categories.append(cat)
                scores.append(stats[cat]['mean_score'])
                counts.append(stats[cat]['count'])
        
        # Plot
        x_pos = np.arange(len(categories))
        bars = plt.bar(x_pos, scores, color=colors[:len(categories)])
        
        # Add count labels on top of bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'n={count}', ha='center', va='bottom')
        
        plt.xticks(x_pos, categories, rotation=45, ha='right')
        plt.ylabel('Mean Score')
        plt.title(f'{block.capitalize()} Images: Score by Consumption Category')
        plt.ylim(0, 5)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Block consumption analysis plot saved: {output_path}')

def perform_statistical_tests(df, group_by):
    """Perform statistical tests to compare groups."""
    # Get groups
    groups = [group['score'].values for name, group in df.groupby(group_by) if len(group) > 0 and name != 'Unknown']
    group_names = [name for name, group in df.groupby(group_by) if len(group) > 0 and name != 'Unknown']
    
    if len(groups) < 2:
        return None, None, []
    
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

def generate_report(consumption_stats, block_stats, anova_result, output_path):
    """Generate analysis report."""
    report = []
    report.append("=" * 80)
    report.append("MONTHLY CONSUMPTION EFFECT ON AESTHETIC JUDGMENT ANALYSIS REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Summary statistics by consumption category
    report.append("SUMMARY STATISTICS BY CONSUMPTION CATEGORY")
    report.append("-" * 80)
    report.append(f"{'Consumption Category':<30} {'N':>5} {'Mean Score':>12} {'Std Score':>12} {'Mean RT (ms)':>15}")
    report.append("-" * 80)
    
    for group, stat in consumption_stats.items():
        report.append(f"{group:<30} {stat['count']:>5} {stat['mean_score']:>12.2f} {stat['std_score']:>12.2f} {stat['mean_reaction_time']:>15.1f}")
    
    report.append("")
    
    # Summary statistics by block type
    report.append("SUMMARY STATISTICS BY BLOCK TYPE AND CONSUMPTION")
    report.append("-" * 80)
    
    for block, stats in block_stats.items():
        report.append(f"\n{block.capitalize()} Images:")
        report.append(f"{'Consumption Category':<30} {'N':>5} {'Mean Score':>12} {'Std Score':>12}")
        report.append("-" * 80)
        
        for group, stat in stats.items():
            report.append(f"{group:<30} {stat['count']:>5} {stat['mean_score']:>12.2f} {stat['std_score']:>12.2f}")
    
    report.append("")
    
    # Statistical tests
    report.append("STATISTICAL TESTS")
    report.append("-" * 80)
    if anova_result[0] is not None:
        report.append(f"Consumption Category ANOVA F-statistic: {anova_result[0]:.4f}")
        report.append(f"Consumption Category ANOVA p-value: {anova_result[1]:.4f}")
    report.append("")
    
    # Key findings
    report.append("KEY FINDINGS")
    report.append("-" * 80)
    
    # Analyze mean scores by consumption category
    consumption_mean_scores = {group: stat['mean_score'] for group, stat in consumption_stats.items()}
    sorted_consumption = sorted(consumption_mean_scores.items(), key=lambda x: x[1], reverse=True)
    
    report.append("\nScore Analysis by Consumption Category:")
    for group, mean_score in sorted_consumption:
        report.append(f"  {group}: {mean_score:.2f}")
    
    # Analyze reaction times
    consumption_mean_rt = {group: stat['mean_reaction_time'] for group, stat in consumption_stats.items()}
    sorted_consumption_rt = sorted(consumption_mean_rt.items(), key=lambda x: x[1])
    
    report.append("\nReaction Time Analysis by Consumption Category:")
    for group, rt in sorted_consumption_rt:
        report.append(f"  {group}: {rt:.1f} ms")
    
    # Interpretation
    report.append("\nINTERPRETATION")
    report.append("-" * 80)
    
    if anova_result[0] is not None and anova_result[1] < 0.05:
        report.append("\nStatistically significant differences found between consumption categories.")
    else:
        report.append("\nNo statistically significant differences found between consumption categories.")
    
    # Additional insights
    report.append("\nADDITIONAL INSIGHTS")
    report.append("-" * 80)
    report.append("\nKey Question Analysis:")
    report.append("1. Does monthly consumption level affect image aesthetic judgments?")
    
    if consumption_mean_scores:
        scores = list(consumption_mean_scores.values())
        if max(scores) - min(scores) > 0.1:
            report.append("   - Yes, monthly consumption level appears to affect aesthetic judgments")
            highest_group = max(consumption_mean_scores, key=consumption_mean_scores.get)
            lowest_group = min(consumption_mean_scores, key=consumption_mean_scores.get)
            report.append(f"   - Highest scoring group: {highest_group} ({consumption_mean_scores[highest_group]:.2f})")
            report.append(f"   - Lowest scoring group: {lowest_group} ({consumption_mean_scores[lowest_group]:.2f})")
        else:
            report.append("   - No significant effect of monthly consumption on aesthetic judgments")
    
    report.append("\nPossible Explanations:")
    report.append("  - Higher consumption may correlate with exposure to diverse aesthetic experiences")
    report.append("  - Different consumption levels may reflect different lifestyles and aesthetic preferences")
    report.append("  - Economic factors may influence how individuals evaluate visual stimuli")
    
    report.append("")
    report.append("=" * 80)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f'Analysis report saved: {output_path}')

def main():
    """Main function to run consumption analysis."""
    print("Reading data...")
    df = read_data()
    print(f"Total records: {len(df)}")
    
    print("\nAnalyzing consumption effects...")
    consumption_stats, block_stats, df = analyze_consumption_effect(df)
    
    print("\nGenerating visualizations...")
    os.makedirs('code/picture', exist_ok=True)
    
    # Generate plots
    plot_consumption_analysis(consumption_stats, 'code/picture/consumption_analysis.png')
    plot_reaction_time_comparison(consumption_stats, 'code/picture/consumption_reaction_time.png')
    plot_score_distribution(consumption_stats, 'Score Distribution by Consumption Category', 'code/picture/consumption_score_distribution.png')
    plot_block_consumption_analysis(block_stats, 'code/picture/consumption_block_analysis.png')
    
    # Perform statistical tests
    print("\nPerforming statistical tests...")
    anova_result = perform_statistical_tests(df, 'consumption_category')
    
    # Generate report
    generate_report(consumption_stats, block_stats, anova_result, 'code/consumption_analysis_report.txt')
    
    print("\n" + "=" * 80)
    print("MONTHLY CONSUMPTION EFFECT ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - code/picture/consumption_analysis.png")
    print("  - code/picture/consumption_reaction_time.png")
    print("  - code/picture/consumption_score_distribution.png")
    print("  - code/picture/consumption_block_analysis.png")
    print("  - code/consumption_analysis_report.txt")

if __name__ == '__main__':
    main()
