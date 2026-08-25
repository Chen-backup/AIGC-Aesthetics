import matplotlib.pyplot as plt
import numpy as np
import os
import csv
import pandas as pd
from scipy import stats

# Set font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

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
                    # Extract relevant features
                    try:
                        record = {
                            'score': int(row.get('score', 0)),
                            'block': row.get('block', ''),
                            'reaction_time': int(row.get('reaction_time_ms', 0)),
                            'user_gender': row.get('user_gender', ''),
                            'user_age': int(row.get('user_age', 0)),
                            'image_url': row.get('image_url', ''),
                        }
                        # Include all images
                        data.append(record)
                    except (ValueError, KeyError):
                        # Skip records with invalid data
                        continue
    
    return pd.DataFrame(data)

def analyze_gender_age_effect(df):
    """Analyze the effect of gender and age on portrait aesthetic judgments."""
    # Process age groups
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
    
    # Extract image gender from filename
    def get_image_gender(image_url):
        # Simple heuristic: check if filename contains 'male' or 'female' indicators
        # For this dataset, we'll create a simplified categorization
        # Note: This is a placeholder - actual implementation depends on dataset structure
        # For now, we'll assume a mix of male and female portraits
        return 'male' if 'male' in image_url.lower() else 'female' if 'female' in image_url.lower() else 'unknown'
    
    df['image_gender'] = df['image_url'].apply(get_image_gender)
    
    # Group by user gender and image gender (for异性相吸分析)
    gender_interaction_grouped = df.groupby(['user_gender', 'image_gender'])
    gender_interaction_stats = {}
    for (user_gender, image_gender), data in gender_interaction_grouped:
        gender_interaction_stats[(user_gender, image_gender)] = {
            'count': len(data),
            'mean_score': data['score'].mean(),
            'std_score': data['score'].std(),
            'mean_reaction_time': data['reaction_time'].mean(),
            'std_reaction_time': data['reaction_time'].std(),
            'score_distribution': data['score'].value_counts().sort_index(),
        }
    
    # Group by age group
    age_grouped = df.groupby('age_group')
    age_stats = {}
    for group, data in age_grouped:
        age_stats[group] = {
            'count': len(data),
            'mean_score': data['score'].mean(),
            'std_score': data['score'].std(),
            'mean_reaction_time': data['reaction_time'].mean(),
            'std_reaction_time': data['reaction_time'].std(),
            'score_distribution': data['score'].value_counts().sort_index(),
        }
    
    # Group by user gender
    user_gender_grouped = df.groupby('user_gender')
    user_gender_stats = {}
    for group, data in user_gender_grouped:
        user_gender_stats[group] = {
            'count': len(data),
            'mean_score': data['score'].mean(),
            'std_score': data['score'].std(),
            'mean_reaction_time': data['reaction_time'].mean(),
            'std_reaction_time': data['reaction_time'].std(),
            'score_distribution': data['score'].value_counts().sort_index(),
        }
    
    return gender_interaction_stats, age_stats, user_gender_stats, df

def plot_gender_interaction_analysis(stats, output_path):
    """Plot gender interaction analysis."""
    plt.figure(figsize=(12, 8))
    
    # Define colors
    colors = {'male': '#9EB5FF', 'female': '#FFB6C1'}
    
    # Prepare data
    categories = []
    scores = []
    counts = []
    
    for (user_gender, image_gender), stat in stats.items():
        if user_gender and image_gender and image_gender != 'unknown':
            categories.append(f"{user_gender} → {image_gender}")
            scores.append(stat['mean_score'])
            counts.append(stat['count'])
    
    # Plot
    x_pos = np.arange(len(categories))
    bars = plt.bar(x_pos, scores, color=[colors.get(cat.split(' → ')[1], '#87CEEB') for cat in categories])
    
    # Add count labels on top of bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'n={count}', ha='center', va='bottom')
    
    plt.xticks(x_pos, categories, rotation=45, ha='right')
    plt.xlabel('User Gender → Image Gender', fontsize=14)
    plt.ylabel('Mean Score', fontsize=14)
    plt.title('Gender Interaction in Portrait Aesthetic Judgment', fontsize=16, fontweight='bold')
    plt.ylim(0, 5)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Gender interaction analysis plot saved: {output_path}')

def plot_age_distribution_analysis(stats, output_path):
    """Plot age distribution analysis."""
    plt.figure(figsize=(12, 8))
    
    # Define colors
    colors = ['#9EB5FF', '#FF9E9E', '#87CEEB', '#FFB6C1', '#90EE90']
    
    # Prepare data
    age_groups = list(stats.keys())
    mean_scores = [stat['mean_score'] for stat in stats.values()]
    counts = [stat['count'] for stat in stats.values()]
    
    # Plot
    x_pos = np.arange(len(age_groups))
    bars = plt.bar(x_pos, mean_scores, color=colors[:len(age_groups)])
    
    # Add count labels on top of bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'n={count}', ha='center', va='bottom')
    
    plt.xticks(x_pos, age_groups)
    plt.xlabel('Age Group', fontsize=14)
    plt.ylabel('Mean Score', fontsize=14)
    plt.title('Age Group Differences in Portrait Aesthetic Judgment', fontsize=16, fontweight='bold')
    plt.ylim(0, 5)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Age distribution analysis plot saved: {output_path}')

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

def generate_report(gender_interaction_stats, age_stats, user_gender_stats, gender_anova, age_anova, output_path):
    """Generate analysis report."""
    report = []
    report.append("=" * 80)
    report.append("GENDER AND AGE EFFECTS IN PORTRAIT AESTHETICS ANALYSIS REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Summary statistics by gender interaction
    report.append("SUMMARY STATISTICS BY GENDER INTERACTION")
    report.append("-" * 80)
    report.append(f"{'User → Image Gender':<25} {'N':>5} {'Mean Score':>12} {'Std Score':>12} {'Mean RT (ms)':>15}")
    report.append("-" * 80)
    
    for (user_gender, image_gender), stat in gender_interaction_stats.items():
        if user_gender and image_gender and image_gender != 'unknown':
            report.append(f"{user_gender} → {image_gender:<15} {stat['count']:>5} {stat['mean_score']:>12.2f} {stat['std_score']:>12.2f} {stat['mean_reaction_time']:>15.1f}")
    
    report.append("")
    
    # Summary statistics by age group
    report.append("SUMMARY STATISTICS BY AGE GROUP")
    report.append("-" * 80)
    report.append(f"{'Age Group':<15} {'N':>5} {'Mean Score':>12} {'Std Score':>12} {'Mean RT (ms)':>15}")
    report.append("-" * 80)
    
    for group, stat in age_stats.items():
        report.append(f"{group:<15} {stat['count']:>5} {stat['mean_score']:>12.2f} {stat['std_score']:>12.2f} {stat['mean_reaction_time']:>15.1f}")
    
    report.append("")
    
    # Summary statistics by user gender
    report.append("SUMMARY STATISTICS BY USER GENDER")
    report.append("-" * 80)
    report.append(f"{'User Gender':<15} {'N':>5} {'Mean Score':>12} {'Std Score':>12} {'Mean RT (ms)':>15}")
    report.append("-" * 80)
    
    for group, stat in user_gender_stats.items():
        report.append(f"{group:<15} {stat['count']:>5} {stat['mean_score']:>12.2f} {stat['std_score']:>12.2f} {stat['mean_reaction_time']:>15.1f}")
    
    report.append("")
    
    # Statistical tests
    report.append("STATISTICAL TESTS")
    report.append("-" * 80)
    if gender_anova[0] is not None:
        report.append(f"Gender Interaction ANOVA F-statistic: {gender_anova[0]:.4f}")
        report.append(f"Gender Interaction ANOVA p-value: {gender_anova[1]:.4f}")
    if age_anova[0] is not None:
        report.append(f"Age Group ANOVA F-statistic: {age_anova[0]:.4f}")
        report.append(f"Age Group ANOVA p-value: {age_anova[1]:.4f}")
    report.append("")
    
    # Key findings
    report.append("KEY FINDINGS")
    report.append("-" * 80)
    
    # Analyze gender interaction
    gender_pairs = []
    for (user_gender, image_gender), stat in gender_interaction_stats.items():
        if user_gender and image_gender and image_gender != 'unknown':
            gender_pairs.append((f"{user_gender} → {image_gender}", stat['mean_score']))
    
    sorted_gender_pairs = sorted(gender_pairs, key=lambda x: x[1], reverse=True)
    report.append("\nGender Interaction Analysis:")
    for pair, score in sorted_gender_pairs:
        report.append(f"  {pair}: {score:.2f}")
    
    # Analyze age groups
    age_mean_scores = {group: stat['mean_score'] for group, stat in age_stats.items()}
    sorted_age = sorted(age_mean_scores.items(), key=lambda x: x[1], reverse=True)
    
    report.append("\nAge Group Analysis:")
    for group, mean_score in sorted_age:
        report.append(f"  {group}: {mean_score:.2f}")
    
    # Interpretation
    report.append("\nINTERPRETATION")
    report.append("-" * 80)
    
    # Analyze异性相吸或同性相斥
    report.append("\nGender Interaction Patterns:")
    same_gender_scores = []
    different_gender_scores = []
    
    for (user_gender, image_gender), stat in gender_interaction_stats.items():
        if user_gender and image_gender and image_gender != 'unknown':
            if user_gender == image_gender:
                same_gender_scores.append(stat['mean_score'])
            else:
                different_gender_scores.append(stat['mean_score'])
    
    if same_gender_scores and different_gender_scores:
        avg_same = sum(same_gender_scores) / len(same_gender_scores)
        avg_diff = sum(different_gender_scores) / len(different_gender_scores)
        report.append(f"  Average score for same gender: {avg_same:.2f}")
        report.append(f"  Average score for different gender: {avg_diff:.2f}")
        
        if avg_diff > avg_same:
            report.append("  → Evidence of 'opposite gender attraction' pattern")
        elif avg_same > avg_diff:
            report.append("  → Evidence of 'same gender preference' pattern")
        else:
            report.append("  → No clear gender preference pattern")
    
    # Analyze age differences
    report.append("\nAge Group Patterns:")
    if len(age_stats) > 1:
        min_age_group = min(age_mean_scores, key=age_mean_scores.get)
        max_age_group = max(age_mean_scores, key=age_mean_scores.get)
        report.append(f"  Highest scoring age group: {max_age_group} ({age_mean_scores[max_age_group]:.2f})")
        report.append(f"  Lowest scoring age group: {min_age_group} ({age_mean_scores[min_age_group]:.2f})")
        report.append(f"  Score difference: {abs(age_mean_scores[max_age_group] - age_mean_scores[min_age_group]):.2f}")
    
    # Statistical significance
    if gender_anova[0] is not None and gender_anova[1] < 0.05:
        report.append("\nStatistically significant differences found between gender interaction groups.")
    else:
        report.append("\nNo statistically significant differences found between gender interaction groups.")
    
    if age_anova[0] is not None and age_anova[1] < 0.05:
        report.append("Statistically significant differences found between age groups.")
    else:
        report.append("No statistically significant differences found between age groups.")
    
    # Additional insights
    report.append("\nADDITIONAL INSIGHTS")
    report.append("-" * 80)
    report.append("\nKey Questions Analysis:")
    report.append("1. Do men and women show 'opposite gender attraction' or 'same gender repulsion' in portrait aesthetic judgments?")
    if same_gender_scores and different_gender_scores:
        if avg_diff > avg_same:
            report.append(f"   - Yes, evidence of opposite gender attraction (different: {avg_diff:.2f} vs same: {avg_same:.2f})")
        elif avg_same > avg_diff:
            report.append(f"   - No, evidence of same gender preference (same: {avg_same:.2f} vs different: {avg_diff:.2f})")
        else:
            report.append("   - No clear pattern observed")
    
    report.append("2. Do different age groups show different tolerance levels for specific styles?")
    if len(age_stats) > 1:
        age_scores = list(age_mean_scores.values())
        if max(age_scores) - min(age_scores) > 0.1:
            report.append("   - Yes, age groups show different tolerance levels")
        else:
            report.append("   - No significant differences in tolerance levels across age groups")
    
    report.append("")
    report.append("=" * 80)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f'Analysis report saved: {output_path}')

def main():
    """Main function to run gender and age analysis."""
    print("Reading data...")
    df = read_data()
    print(f"Total portrait records: {len(df)}")
    
    if len(df) == 0:
        print("\nNo portrait data found. Please check your dataset.")
        return
    
    print("\nAnalyzing gender and age effects...")
    try:
        gender_interaction_stats, age_stats, user_gender_stats, df = analyze_gender_age_effect(df)
    except KeyError as e:
        print(f"\nError: {e}. Please check if the required columns exist in your data.")
        return
    
    print("\nGenerating visualizations...")
    os.makedirs('code/picture', exist_ok=True)
    
    # Generate plots if data is available
    if gender_interaction_stats:
        plot_gender_interaction_analysis(gender_interaction_stats, 'code/picture/gender_interaction_analysis.png')
    if age_stats:
        plot_age_distribution_analysis(age_stats, 'code/picture/age_distribution_analysis.png')
    if user_gender_stats:
        plot_score_distribution(user_gender_stats, 'Score Distribution by User Gender', 'code/picture/gender_score_distribution.png')
    if age_stats:
        plot_reaction_time_comparison(age_stats, 'Reaction Time by Age Group', 'code/picture/age_reaction_time.png')
    
    # Create gender interaction grouping for ANOVA
    def get_gender_interaction(row):
        if 'user_gender' in row and 'image_gender' in row and row['image_gender'] != 'unknown':
            return f"{row['user_gender']}→{row['image_gender']}"
        return 'unknown'
    
    try:
        df['gender_interaction'] = df.apply(get_gender_interaction, axis=1)
    except Exception as e:
        print(f"\nError creating gender interaction column: {e}")
        return
    
    # Perform statistical tests
    print("\nPerforming statistical tests...")
    try:
        gender_anova = perform_statistical_tests(df[df['gender_interaction'] != 'unknown'], 'gender_interaction')
        age_anova = perform_statistical_tests(df, 'age_group')
    except Exception as e:
        print(f"\nError performing statistical tests: {e}")
        return
    
    # Generate report
    try:
        generate_report(gender_interaction_stats, age_stats, user_gender_stats, gender_anova, age_anova, 'code/gender_age_portrait_analysis_report.txt')
    except Exception as e:
        print(f"\nError generating report: {e}")
        return
    
    print("\n" + "=" * 80)
    print("GENDER AND AGE EFFECTS IN PORTRAIT AESTHETICS ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - code/picture/gender_interaction_analysis.png")
    print("  - code/picture/age_distribution_analysis.png")
    print("  - code/picture/gender_score_distribution.png")
    print("  - code/picture/age_reaction_time.png")
    print("  - code/gender_age_portrait_analysis_report.txt")

if __name__ == '__main__':
    main()
