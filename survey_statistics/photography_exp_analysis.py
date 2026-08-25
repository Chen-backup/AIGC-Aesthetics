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
                            'user_photo_exp': row.get('user_photo_exp', ''),
                        }
                        data.append(record)
                    except (ValueError, KeyError):
                        # Skip records with invalid data
                        continue
    
    return pd.DataFrame(data)

def analyze_photo_exp_effect(df):
    """Analyze the effect of photography experience on aesthetic judgments."""
    # Process photography experience levels
    def categorize_photo_exp(exp):
        if not exp:
            return 'Unknown'
        # Map photography experience categories
        exp_map = {
            '无': 'No Experience',
            '手机随拍': 'Mobile Photography',
            '业余爱好': 'Amateur',
            '专业摄影师': 'Professional'
        }
        return exp_map.get(exp, 'Unknown')
    
    df['photo_exp_category'] = df['user_photo_exp'].apply(categorize_photo_exp)
    
    # Group by photography experience category
    photo_exp_grouped = df.groupby('photo_exp_category')
    photo_exp_stats = {}
    for group, data in photo_exp_grouped:
        if group != 'Unknown':
            photo_exp_stats[group] = {
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
            block_exp_grouped = block_data.groupby('photo_exp_category')
            block_exp_stats = {}
            for group, data in block_exp_grouped:
                if group != 'Unknown':
                    block_exp_stats[group] = {
                        'count': len(data),
                        'mean_score': data['score'].mean(),
                        'std_score': data['score'].std(),
                    }
            block_stats[block] = block_exp_stats
    
    return photo_exp_stats, block_stats, df

def plot_photo_exp_analysis(stats, output_path):
    """Plot photography experience analysis."""
    plt.figure(figsize=(12, 8))
    
    # Define colors
    colors = ['#9EB5FF', '#FF9E9E', '#87CEEB', '#FFB6C1']
    
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
    plt.xlabel('Photography Experience Level', fontsize=14)
    plt.ylabel('Mean Score', fontsize=14)
    plt.title('Effect of Photography Experience on Aesthetic Judgment', fontsize=16, fontweight='bold')
    plt.ylim(0, 5)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Photography experience analysis plot saved: {output_path}')

def plot_reaction_time_comparison(stats, output_path):
    """Plot reaction time comparison by photography experience category."""
    plt.figure(figsize=(10, 6))
    
    # Define colors
    colors = ['#9EB5FF', '#FF9E9E', '#87CEEB', '#FFB6C1']
    
    # Prepare data
    groups = list(stats.keys())
    mean_rt = [stat['mean_reaction_time'] for stat in stats.values()]
    std_rt = [stat['std_reaction_time'] for stat in stats.values()]
    
    # Plot
    x_pos = np.arange(len(groups))
    plt.bar(x_pos, mean_rt, yerr=std_rt, capsize=5, color=[colors[i % len(colors)] for i in range(len(groups))])
    plt.xticks(x_pos, groups, rotation=45, ha='right')
    plt.xlabel('Photography Experience Level', fontsize=14)
    plt.ylabel('Mean Reaction Time (ms)', fontsize=14)
    plt.title('Reaction Time by Photography Experience Level', fontsize=16, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Reaction time comparison plot saved: {output_path}')

def plot_score_distribution(stats, title, output_path):
    """Plot score distribution by photography experience category."""
    plt.figure(figsize=(12, 8))
    
    # Define colors
    colors = ['#9EB5FF', '#FF9E9E', '#87CEEB', '#FFB6C1']
    
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

def plot_block_photo_exp_analysis(block_stats, output_path):
    """Plot photography experience analysis by block type."""
    plt.figure(figsize=(14, 10))
    
    # Define colors
    colors = ['#9EB5FF', '#FF9E9E', '#87CEEB', '#FFB6C1']
    
    # Get all unique photography experience categories
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
        plt.title(f'{block.capitalize()} Images: Score by Photography Experience')
        plt.ylim(0, 5)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Block photography experience analysis plot saved: {output_path}')

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

def generate_report(photo_exp_stats, block_stats, anova_result, output_path):
    """Generate analysis report."""
    report = []
    report.append("=" * 80)
    report.append("PHOTOGRAPHY EXPERIENCE EFFECT ON AESTHETIC JUDGMENT ANALYSIS REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Summary statistics by photography experience category
    report.append("SUMMARY STATISTICS BY PHOTOGRAPHY EXPERIENCE")
    report.append("-" * 80)
    report.append(f"{'Experience Level':<30} {'N':>5} {'Mean Score':>12} {'Std Score':>12} {'Mean RT (ms)':>15}")
    report.append("-" * 80)
    
    for group, stat in photo_exp_stats.items():
        report.append(f"{group:<30} {stat['count']:>5} {stat['mean_score']:>12.2f} {stat['std_score']:>12.2f} {stat['mean_reaction_time']:>15.1f}")
    
    report.append("")
    
    # Summary statistics by block type
    report.append("SUMMARY STATISTICS BY BLOCK TYPE AND PHOTOGRAPHY EXPERIENCE")
    report.append("-" * 80)
    
    for block, stats in block_stats.items():
        report.append(f"\n{block.capitalize()} Images:")
        report.append(f"{'Experience Level':<30} {'N':>5} {'Mean Score':>12} {'Std Score':>12}")
        report.append("-" * 80)
        
        for group, stat in stats.items():
            report.append(f"{group:<30} {stat['count']:>5} {stat['mean_score']:>12.2f} {stat['std_score']:>12.2f}")
    
    report.append("")
    
    # Statistical tests
    report.append("STATISTICAL TESTS")
    report.append("-" * 80)
    if anova_result[0] is not None:
        report.append(f"Photography Experience ANOVA F-statistic: {anova_result[0]:.4f}")
        report.append(f"Photography Experience ANOVA p-value: {anova_result[1]:.4f}")
    report.append("")
    
    # Key findings
    report.append("KEY FINDINGS")
    report.append("-" * 80)
    
    # Analyze mean scores by photography experience category
    exp_mean_scores = {group: stat['mean_score'] for group, stat in photo_exp_stats.items()}
    sorted_exp = sorted(exp_mean_scores.items(), key=lambda x: x[1], reverse=True)
    
    report.append("\nScore Analysis by Photography Experience:")
    for group, mean_score in sorted_exp:
        report.append(f"  {group}: {mean_score:.2f}")
    
    # Analyze reaction times
    exp_mean_rt = {group: stat['mean_reaction_time'] for group, stat in photo_exp_stats.items()}
    sorted_exp_rt = sorted(exp_mean_rt.items(), key=lambda x: x[1])
    
    report.append("\nReaction Time Analysis by Photography Experience:")
    for group, rt in sorted_exp_rt:
        report.append(f"  {group}: {rt:.1f} ms")
    
    # Interpretation
    report.append("\nINTERPRETATION")
    report.append("-" * 80)
    
    if anova_result[0] is not None and anova_result[1] < 0.05:
        report.append("\nStatistically significant differences found between photography experience groups.")
    else:
        report.append("\nNo statistically significant differences found between photography experience groups.")
    
    # Additional insights
    report.append("\nADDITIONAL INSIGHTS")
    report.append("-" * 80)
    report.append("\nKey Question Analysis:")
    report.append("1. Does photography experience affect image aesthetic judgments?")
    
    if exp_mean_scores:
        scores = list(exp_mean_scores.values())
        if max(scores) - min(scores) > 0.1:
            report.append("   - Yes, photography experience appears to affect aesthetic judgments")
            highest_group = max(exp_mean_scores, key=exp_mean_scores.get)
            lowest_group = min(exp_mean_scores, key=exp_mean_scores.get)
            report.append(f"   - Highest scoring group: {highest_group} ({exp_mean_scores[highest_group]:.2f})")
            report.append(f"   - Lowest scoring group: {lowest_group} ({exp_mean_scores[lowest_group]:.2f})")
        else:
            report.append("   - No significant effect of photography experience on aesthetic judgments")
    
    report.append("\nPossible Explanations:")
    report.append("  - Photography experience may develop a more discerning eye for visual composition")
    report.append("  - Different experience levels may have different aesthetic standards")
    report.append("  - Technical knowledge of photography may influence how individuals evaluate images")
    
    report.append("")
    report.append("=" * 80)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f'Analysis report saved: {output_path}')

def main():
    """Main function to run photography experience analysis."""
    print("Reading data...")
    df = read_data()
    print(f"Total records: {len(df)}")
    
    print("\nAnalyzing photography experience effects...")
    photo_exp_stats, block_stats, df = analyze_photo_exp_effect(df)
    
    print("\nGenerating visualizations...")
    os.makedirs('code/picture', exist_ok=True)
    
    # Generate plots
    plot_photo_exp_analysis(photo_exp_stats, 'code/picture/photo_exp_analysis.png')
    plot_reaction_time_comparison(photo_exp_stats, 'code/picture/photo_exp_reaction_time.png')
    plot_score_distribution(photo_exp_stats, 'Score Distribution by Photography Experience', 'code/picture/photo_exp_score_distribution.png')
    plot_block_photo_exp_analysis(block_stats, 'code/picture/photo_exp_block_analysis.png')
    
    # Perform statistical tests
    print("\nPerforming statistical tests...")
    anova_result = perform_statistical_tests(df, 'photo_exp_category')
    
    # Generate report
    generate_report(photo_exp_stats, block_stats, anova_result, 'code/photo_exp_analysis_report.txt')
    
    print("\n" + "=" * 80)
    print("PHOTOGRAPHY EXPERIENCE EFFECT ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - code/picture/photo_exp_analysis.png")
    print("  - code/picture/photo_exp_reaction_time.png")
    print("  - code/picture/photo_exp_score_distribution.png")
    print("  - code/picture/photo_exp_block_analysis.png")
    print("  - code/photo_exp_analysis_report.txt")

if __name__ == '__main__':
    main()
