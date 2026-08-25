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
                    record = {
                        'score': int(row.get('score', 0)),
                        'block': row.get('block', ''),
                        'reaction_time': int(row.get('reaction_time_ms', 0)),
                        'user_video_time': row.get('user_video_time', ''),
                        'user_platforms': row.get('user_platforms', ''),
                        'user_content_types': row.get('user_content_types', ''),
                    }
                    data.append(record)
    
    return pd.DataFrame(data)

def analyze_digital_native_effect(df):
    """Analyze the effect of digital native factors on aesthetic judgments."""
    # Process video time
    def categorize_video_time(time_str):
        if '4h' in time_str or '4-5' in time_str or '5h' in time_str or '>5' in time_str:
            return 'High (4h+)'
        elif '2-3' in time_str:
            return 'Medium (2-3h)'
        elif '1-2' in time_str:
            return 'Low (1-2h)'
        else:
            return 'Very Low (<1h)'
    
    df['video_time_category'] = df['user_video_time'].apply(categorize_video_time)
    
    # Process platforms
    def has_xiaohongshu(platforms):
        return '小红书' in platforms if platforms else False
    
    def has_douyin(platforms):
        return '抖音' in platforms or '快手' in platforms if platforms else False
    
    df['uses_xiaohongshu'] = df['user_platforms'].apply(has_xiaohongshu)
    df['uses_douyin_kuaishou'] = df['user_platforms'].apply(has_douyin)
    
    # Process content types
    content_categories = {
        'entertainment': ['娱乐', '动漫', '游戏', '影视', '音乐'],
        'education': ['学习', '科普', '知识'],
        'lifestyle': ['生活', '时尚', '美食', '旅行'],
        'technology': ['科技', '数码', 'AI'],
    }
    
    for category, keywords in content_categories.items():
        def has_category(content):
            if not content:
                return False
            for keyword in keywords:
                if keyword in content:
                    return True
            return False
        df[f'prefers_{category}'] = df['user_content_types'].apply(has_category)
    
    # Group by video time
    video_time_grouped = df.groupby('video_time_category')
    video_time_stats = {}
    for group, data in video_time_grouped:
        video_time_stats[group] = {
            'count': len(data),
            'mean_score': data['score'].mean(),
            'std_score': data['score'].std(),
            'mean_reaction_time': data['reaction_time'].mean(),
            'std_reaction_time': data['reaction_time'].std(),
            'score_distribution': data['score'].value_counts().sort_index(),
        }
    
    # Group by platform
    platform_stats = {}
    # Xiaohongshu users
    xhs_users = df[df['uses_xiaohongshu']]
    platform_stats['Xiaohongshu Users'] = {
        'count': len(xhs_users),
        'mean_score': xhs_users['score'].mean(),
        'std_score': xhs_users['score'].std(),
        'mean_reaction_time': xhs_users['reaction_time'].mean(),
        'std_reaction_time': xhs_users['reaction_time'].std(),
        'score_distribution': xhs_users['score'].value_counts().sort_index(),
    }
    
    # Douyin/Kuaishou users
    douyin_users = df[df['uses_douyin_kuaishou']]
    platform_stats['Douyin/Kuaishou Users'] = {
        'count': len(douyin_users),
        'mean_score': douyin_users['score'].mean(),
        'std_score': douyin_users['score'].std(),
        'mean_reaction_time': douyin_users['reaction_time'].mean(),
        'std_reaction_time': douyin_users['reaction_time'].std(),
        'score_distribution': douyin_users['score'].value_counts().sort_index(),
    }
    
    # Both platforms
    both_users = df[df['uses_xiaohongshu'] & df['uses_douyin_kuaishou']]
    platform_stats['Both Platforms'] = {
        'count': len(both_users),
        'mean_score': both_users['score'].mean(),
        'std_score': both_users['score'].std(),
        'mean_reaction_time': both_users['reaction_time'].mean(),
        'std_reaction_time': both_users['reaction_time'].std(),
        'score_distribution': both_users['score'].value_counts().sort_index(),
    }
    
    # Neither platform
    neither_users = df[~df['uses_xiaohongshu'] & ~df['uses_douyin_kuaishou']]
    platform_stats['Neither Platform'] = {
        'count': len(neither_users),
        'mean_score': neither_users['score'].mean(),
        'std_score': neither_users['score'].std(),
        'mean_reaction_time': neither_users['reaction_time'].mean(),
        'std_reaction_time': neither_users['reaction_time'].std(),
        'score_distribution': neither_users['score'].value_counts().sort_index(),
    }
    
    # Group by content type
    content_stats = {}
    for category in content_categories.keys():
        content_users = df[df[f'prefers_{category}']]
        content_stats[category] = {
            'count': len(content_users),
            'mean_score': content_users['score'].mean(),
            'std_score': content_users['score'].std(),
            'mean_reaction_time': content_users['reaction_time'].mean(),
            'std_reaction_time': content_users['reaction_time'].std(),
        }
    
    return video_time_stats, platform_stats, content_stats, df

def plot_score_distribution(stats, title, output_path):
    """Plot score distribution by category."""
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

def plot_score_boxplot(df, group_by, title, output_path):
    """Plot boxplot of scores by category."""
    plt.figure(figsize=(10, 6))
    
    # Define colors
    colors = ['#9EB5FF', '#FF9E9E', '#87CEEB', '#FFB6C1', '#90EE90']
    
    # Create boxplot
    boxplot_data = [group['score'].values for name, group in df.groupby(group_by)]
    boxplot_labels = [name for name, group in df.groupby(group_by)]
    
    box = plt.boxplot(boxplot_data, tick_labels=boxplot_labels, patch_artist=True)
    
    # Set colors
    for patch, label, i in zip(box['boxes'], boxplot_labels, range(len(boxplot_labels))):
        patch.set_facecolor(colors[i % len(colors)])
    
    plt.xlabel('Category', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.ylim(0, 8)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Score boxplot saved: {output_path}')

def plot_reaction_time_comparison(stats, title, output_path):
    """Plot reaction time comparison by category."""
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
    plt.xlabel('Category', fontsize=14)
    plt.ylabel('Mean Reaction Time (ms)', fontsize=14)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Reaction time comparison plot saved: {output_path}')

def plot_platform_comparison(platform_stats, output_path):
    """Plot platform comparison."""
    plt.figure(figsize=(12, 8))
    
    # Define colors
    colors = ['#9EB5FF', '#FF9E9E', '#87CEEB', '#FFB6C1']
    
    # Prepare data
    platforms = list(platform_stats.keys())
    mean_scores = [stat['mean_score'] for stat in platform_stats.values()]
    counts = [stat['count'] for stat in platform_stats.values()]
    
    # Plot
    x_pos = np.arange(len(platforms))
    bars = plt.bar(x_pos, mean_scores, color=colors[:len(platforms)])
    
    # Add count labels on top of bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'n={count}', ha='center', va='bottom')
    
    plt.xticks(x_pos, platforms, rotation=45, ha='right')
    plt.xlabel('Platform Usage', fontsize=14)
    plt.ylabel('Mean Score', fontsize=14)
    plt.title('Score by Platform Usage', fontsize=16, fontweight='bold')
    plt.ylim(0, 5)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Platform comparison plot saved: {output_path}')

def plot_content_preference_analysis(content_stats, output_path):
    """Plot content preference analysis."""
    plt.figure(figsize=(12, 8))
    
    # Define colors
    colors = ['#9EB5FF', '#FF9E9E', '#87CEEB', '#FFB6C1']
    
    # Prepare data
    content_types = list(content_stats.keys())
    mean_scores = [stat['mean_score'] for stat in content_stats.values()]
    counts = [stat['count'] for stat in content_stats.values()]
    
    # Plot
    x_pos = np.arange(len(content_types))
    bars = plt.bar(x_pos, mean_scores, color=colors[:len(content_types)])
    
    # Add count labels on top of bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'n={count}', ha='center', va='bottom')
    
    plt.xticks(x_pos, content_types, rotation=45, ha='right')
    plt.xlabel('Content Preference', fontsize=14)
    plt.ylabel('Mean Score', fontsize=14)
    plt.title('Score by Content Preference', fontsize=16, fontweight='bold')
    plt.ylim(0, 5)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Content preference analysis plot saved: {output_path}')

def perform_statistical_tests(df, group_by):
    """Perform statistical tests to compare groups."""
    # Get groups
    groups = [group['score'].values for name, group in df.groupby(group_by) if len(group) > 0]
    group_names = [name for name, group in df.groupby(group_by) if len(group) > 0]
    
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

def generate_report(video_time_stats, platform_stats, content_stats, video_time_anova, platform_anova, content_anova, output_path):
    """Generate analysis report."""
    report = []
    report.append("=" * 80)
    report.append("DIGITAL NATIVE AND SHORT VIDEO ERA AESTHETIC ANALYSIS REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Summary statistics by video time
    report.append("SUMMARY STATISTICS BY VIDEO TIME")
    report.append("-" * 80)
    report.append(f"{'Video Time':<20} {'N':>5} {'Mean Score':>12} {'Std Score':>12} {'Mean RT (ms)':>15}")
    report.append("-" * 80)
    
    for group, stat in video_time_stats.items():
        report.append(f"{group:<20} {stat['count']:>5} {stat['mean_score']:>12.2f} {stat['std_score']:>12.2f} {stat['mean_reaction_time']:>15.1f}")
    
    report.append("")
    
    # Summary statistics by platform
    report.append("SUMMARY STATISTICS BY PLATFORM")
    report.append("-" * 80)
    report.append(f"{'Platform':<20} {'N':>5} {'Mean Score':>12} {'Std Score':>12} {'Mean RT (ms)':>15}")
    report.append("-" * 80)
    
    for group, stat in platform_stats.items():
        report.append(f"{group:<20} {stat['count']:>5} {stat['mean_score']:>12.2f} {stat['std_score']:>12.2f} {stat['mean_reaction_time']:>15.1f}")
    
    report.append("")
    
    # Summary statistics by content type
    report.append("SUMMARY STATISTICS BY CONTENT TYPE")
    report.append("-" * 80)
    report.append(f"{'Content Type':<20} {'N':>5} {'Mean Score':>12} {'Std Score':>12} {'Mean RT (ms)':>15}")
    report.append("-" * 80)
    
    for group, stat in content_stats.items():
        report.append(f"{group:<20} {stat['count']:>5} {stat['mean_score']:>12.2f} {stat['std_score']:>12.2f} {stat['mean_reaction_time']:>15.1f}")
    
    report.append("")
    
    # Statistical tests
    report.append("STATISTICAL TESTS")
    report.append("-" * 80)
    if video_time_anova[0] is not None:
        report.append(f"Video Time ANOVA F-statistic: {video_time_anova[0]:.4f}")
        report.append(f"Video Time ANOVA p-value: {video_time_anova[1]:.4f}")
    if platform_anova[0] is not None:
        report.append(f"Platform ANOVA F-statistic: {platform_anova[0]:.4f}")
        report.append(f"Platform ANOVA p-value: {platform_anova[1]:.4f}")
    if content_anova[0] is not None:
        report.append(f"Content Type ANOVA F-statistic: {content_anova[0]:.4f}")
        report.append(f"Content Type ANOVA p-value: {content_anova[1]:.4f}")
    report.append("")
    
    # Key findings
    report.append("KEY FINDINGS")
    report.append("-" * 80)
    
    # Analyze mean scores by video time
    video_time_mean_scores = {group: stat['mean_score'] for group, stat in video_time_stats.items()}
    sorted_video_time = sorted(video_time_mean_scores.items(), key=lambda x: x[1], reverse=True)
    
    report.append("\nScore Analysis by Video Time:")
    for group, mean_score in sorted_video_time:
        report.append(f"  {group}: {mean_score:.2f}")
    
    # Analyze mean scores by platform
    platform_mean_scores = {group: stat['mean_score'] for group, stat in platform_stats.items()}
    sorted_platform = sorted(platform_mean_scores.items(), key=lambda x: x[1], reverse=True)
    
    report.append("\nScore Analysis by Platform:")
    for group, mean_score in sorted_platform:
        report.append(f"  {group}: {mean_score:.2f}")
    
    # Analyze mean scores by content type
    content_mean_scores = {group: stat['mean_score'] for group, stat in content_stats.items()}
    sorted_content = sorted(content_mean_scores.items(), key=lambda x: x[1], reverse=True)
    
    report.append("\nScore Analysis by Content Type:")
    for group, mean_score in sorted_content:
        report.append(f"  {group}: {mean_score:.2f}")
    
    # Analyze reaction times
    video_time_mean_rt = {group: stat['mean_reaction_time'] for group, stat in video_time_stats.items()}
    sorted_video_time_rt = sorted(video_time_mean_rt.items(), key=lambda x: x[1])
    
    report.append("\nReaction Time Analysis by Video Time:")
    for group, rt in sorted_video_time_rt:
        report.append(f"  {group}: {rt:.1f} ms")
    
    # Interpretation
    report.append("\nINTERPRETATION")
    report.append("-" * 80)
    
    if video_time_anova[0] is not None and video_time_anova[1] < 0.05:
        report.append("\nStatistically significant differences found between video time groups.")
    else:
        report.append("\nNo statistically significant differences found between video time groups.")
    
    if platform_anova[0] is not None and platform_anova[1] < 0.05:
        report.append("Statistically significant differences found between platform groups.")
    else:
        report.append("No statistically significant differences found between platform groups.")
    
    if content_anova[0] is not None and content_anova[1] < 0.05:
        report.append("Statistically significant differences found between content type groups.")
    else:
        report.append("No statistically significant differences found between content type groups.")
    
    # Additional insights
    report.append("\nADDITIONAL INSIGHTS")
    report.append("-" * 80)
    report.append("\nKey Questions Analysis:")
    report.append("1. Do people who spend over 4 hours daily on short videos prefer 'high saturation, strong visual impact' images?")
    if 'High (4h+)' in video_time_stats:
        high_time_score = video_time_stats['High (4h+)']['mean_score']
        low_time_score = video_time_stats['Very Low (<1h)']['mean_score'] if 'Very Low (<1h)' in video_time_stats else 0
        if high_time_score > low_time_score:
            report.append(f"   - Yes, high video time users (mean score: {high_time_score:.2f}) tend to give higher scores than low video time users")
        else:
            report.append(f"   - No clear difference, high video time users (mean score: {high_time_score:.2f}) vs low video time users")
    
    report.append("2. Is there a significant community barrier between Xiaohongshu users and Douyin/Kuaishou users?")
    if 'Xiaohongshu Users' in platform_stats and 'Douyin/Kuaishou Users' in platform_stats:
        xhs_score = platform_stats['Xiaohongshu Users']['mean_score']
        douyin_score = platform_stats['Douyin/Kuaishou Users']['mean_score']
        report.append(f"   - Xiaohongshu users: {xhs_score:.2f}, Douyin/Kuaishou users: {douyin_score:.2f}")
        if abs(xhs_score - douyin_score) > 0.1:
            report.append("   - Significant difference observed, suggesting community barriers")
        else:
            report.append("   - No significant difference, suggesting minimal community barriers")
    
    report.append("3. Does content preference affect aesthetic judgments?")
    content_scores = [stat['mean_score'] for stat in content_stats.values()]
    if max(content_scores) - min(content_scores) > 0.1:
        report.append("   - Yes, content preference appears to influence aesthetic judgments")
    else:
        report.append("   - No clear effect of content preference on aesthetic judgments")
    
    report.append("")
    report.append("=" * 80)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f'Analysis report saved: {output_path}')

def main():
    """Main function to run digital native analysis."""
    print("Reading data...")
    df = read_data()
    print(f"Total records: {len(df)}")
    
    print("\nAnalyzing digital native effects...")
    video_time_stats, platform_stats, content_stats, df = analyze_digital_native_effect(df)
    
    print("\nGenerating visualizations...")
    os.makedirs('code/picture', exist_ok=True)
    
    # Generate plots
    plot_score_distribution(video_time_stats, 'Score Distribution by Video Time', 'code/picture/digital_video_time_score_distribution.png')
    plot_score_distribution(platform_stats, 'Score Distribution by Platform', 'code/picture/digital_platform_score_distribution.png')
    plot_score_boxplot(df, 'video_time_category', 'Score Distribution by Video Time', 'code/picture/digital_video_time_score_boxplot.png')
    plot_reaction_time_comparison(video_time_stats, 'Reaction Time by Video Time', 'code/picture/digital_video_time_reaction_time.png')
    plot_platform_comparison(platform_stats, 'code/picture/digital_platform_comparison.png')
    plot_content_preference_analysis(content_stats, 'code/picture/digital_content_preference_analysis.png')
    
    # Create platform grouping column
    def get_platform_group(row):
        if row['uses_xiaohongshu']:
            return 'Xiaohongshu'
        elif row['uses_douyin_kuaishou']:
            return 'Douyin/Kuaishou'
        else:
            return 'Neither'
    
    df['platform_group'] = df.apply(get_platform_group, axis=1)
    
    # Create content grouping column
    def get_content_group(row):
        for cat in ['entertainment', 'education', 'lifestyle', 'technology']:
            if row[f'prefers_{cat}']:
                return cat
        return 'None'
    
    df['content_group'] = df.apply(get_content_group, axis=1)
    
    # Perform statistical tests
    print("\nPerforming statistical tests...")
    video_time_anova = perform_statistical_tests(df, 'video_time_category')
    platform_anova = perform_statistical_tests(df, 'platform_group')
    content_anova = perform_statistical_tests(df, 'content_group')
    
    # Generate report
    generate_report(video_time_stats, platform_stats, content_stats, video_time_anova, platform_anova, content_anova, 'code/digital_native_analysis_report.txt')
    
    print("\n" + "=" * 80)
    print("DIGITAL NATIVE AND SHORT VIDEO ERA AESTHETIC ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - code/picture/digital_video_time_score_distribution.png")
    print("  - code/picture/digital_platform_score_distribution.png")
    print("  - code/picture/digital_video_time_score_boxplot.png")
    print("  - code/picture/digital_video_time_reaction_time.png")
    print("  - code/picture/digital_platform_comparison.png")
    print("  - code/picture/digital_content_preference_analysis.png")
    print("  - code/digital_native_analysis_report.txt")

if __name__ == '__main__':
    main()
