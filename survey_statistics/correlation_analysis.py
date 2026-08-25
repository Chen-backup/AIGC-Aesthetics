"""
Correlation Analysis for Image Aesthetics Study
This script analyzes correlations between image features, user characteristics,
and aesthetic scores, and presents the results as a heatmap.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import os
import csv
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

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
                        'block': row.get('block', ''),
                        'reaction_time': int(row.get('reaction_time_ms', 0)),
                        'n_changes': int(row.get('n_changes', 0)),
                        'img_width': int(row.get('img_natural_width', 0)),
                        'img_height': int(row.get('img_natural_height', 0)),
                        'user_gender': row.get('user_gender', ''),
                        'user_age': int(row.get('user_age', 0)),
                        'user_education': row.get('user_education', ''),
                        'user_art_training': row.get('user_art_training', ''),
                        'user_photo_exp': row.get('user_photo_exp', ''),
                        'user_growth_place': row.get('user_growth_place', ''),
                        'user_current_place': row.get('user_current_place', ''),
                        'user_video_time': row.get('user_video_time', ''),
                        'user_platforms': row.get('user_platforms', ''),
                    }
                    
                    # Calculate additional features
                    if record['img_width'] > 0 and record['img_height'] > 0:
                        record['aspect_ratio'] = record['img_width'] / record['img_height']
                        record['image_area'] = record['img_width'] * record['img_height']
                    else:
                        record['aspect_ratio'] = 0
                        record['image_area'] = 0
                    
                    # Convert categorical variables to numerical
                    record['gender_code'] = 1 if record['user_gender'] == '男' else 0 if record['user_gender'] == '女' else 2
                    record['art_training_code'] = 1 if '是' in record['user_art_training'] else 0
                    record['growth_place_code'] = 1 if '一线' in record['user_growth_place'] else 0
                    record['current_place_code'] = 1 if '一线' in record['user_current_place'] else 0
                    
                    data.append(record)
    
    return pd.DataFrame(data)


def calculate_correlations(df):
    """Calculate correlation matrix."""
    # Select numerical columns for correlation analysis
    numerical_cols = ['score', 'reaction_time', 'n_changes', 'img_width', 'img_height', 
                     'aspect_ratio', 'image_area', 'user_age', 'gender_code', 
                     'art_training_code', 'growth_place_code', 'current_place_code']
    
    # Create correlation matrix using Spearman (more robust for non-normal data)
    corr_matrix = df[numerical_cols].corr(method='spearman')
    
    # Also calculate Pearson correlation for comparison
    pearson_matrix = df[numerical_cols].corr(method='pearson')
    
    return corr_matrix, pearson_matrix


def plot_heatmap(corr_matrix, title, output_path):
    """Plot correlation heatmap."""
    plt.figure(figsize=(12, 10))
    
    # Create heatmap with annotations
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                square=True, linewidths=.5, vmin=-1, vmax=1)
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Heatmap saved: {output_path}')


def analyze_correlations(df, corr_matrix):
    """Analyze and summarize significant correlations."""
    significant_correlations = []
    
    # Define significance threshold
    significance_threshold = 0.1
    
    # Iterate through correlation matrix
    for i, row in corr_matrix.iterrows():
        for j, corr in enumerate(row):
            col = corr_matrix.columns[j]
            if i != col and abs(corr) >= significance_threshold:
                significant_correlations.append({
                    'feature1': i,
                    'feature2': col,
                    'correlation': corr
                })
    
    # Sort by absolute correlation
    significant_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
    
    return significant_correlations


def generate_correlation_report(significant_correlations, output_path):
    """Generate correlation analysis report."""
    
    report = []
    report.append("=" * 80)
    report.append("CORRELATION ANALYSIS REPORT")
    report.append("=" * 80)
    report.append("")
    report.append("SIGNIFICANT CORRELATIONS (|r| ≥ 0.1)")
    report.append("-" * 80)
    report.append(f"{'Feature 1':<20} {'Feature 2':<20} {'Correlation':<10}")
    report.append("-" * 80)
    
    for item in significant_correlations:
        report.append(f"{item['feature1']:<20} {item['feature2']:<20} {item['correlation']:>10.2f}")
    
    report.append("")
    report.append("-" * 80)
    report.append("KEY FINDINGS")
    report.append("-" * 80)
    
    # Analyze score correlations specifically
    score_correlations = [item for item in significant_correlations if 'score' in [item['feature1'], item['feature2']]]
    
    if score_correlations:
        report.append("\nCorrelations with aesthetic score:")
        for item in score_correlations[:10]:  # Top 10
            feature = item['feature1'] if item['feature2'] == 'score' else item['feature2']
            corr = item['correlation']
            report.append(f"  {feature}: {corr:.2f}")
    
    report.append("")
    report.append("=" * 80)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f'Correlation report saved: {output_path}')


def main():
    """Main function to run correlation analysis."""
    
    print("Reading data...")
    df = read_data()
    print(f"Total records: {len(df)}")
    
    print("\nCalculating correlations...")
    spearman_matrix, pearson_matrix = calculate_correlations(df)
    
    print("\nGenerating heatmaps...")
    os.makedirs('code/picture', exist_ok=True)
    
    # Generate heatmaps
    plot_heatmap(spearman_matrix, 'Spearman Correlation Heatmap', 'code/picture/spearman_correlation.png')
    plot_heatmap(pearson_matrix, 'Pearson Correlation Heatmap', 'code/picture/pearson_correlation.png')
    
    # Analyze significant correlations
    significant_correlations = analyze_correlations(df, spearman_matrix)
    
    # Generate report
    generate_correlation_report(significant_correlations, 'code/correlation_analysis.txt')
    
    print("\n" + "=" * 80)
    print("CORRELATION ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - code/picture/spearman_correlation.png")
    print("  - code/picture/pearson_correlation.png")
    print("  - code/correlation_analysis.txt")


if __name__ == '__main__':
    main()
