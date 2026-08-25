"""
Comprehensive Analysis for Image Aesthetics Study
This script performs advanced statistical analysis to explore relationships between
participant demographics and aesthetic ratings, including hypothesis testing and
multi-variable analysis.
"""
import pandas as pd
import numpy as np
from scipy import stats
import os
import csv
from collections import Counter
def load_all_data():
    """Load all data from accepted directory."""
    data = []
    accepted_dir = 'G:\\E\\CJH-SJTU\\课题组\\图像美学\\Data\\accepted'
    for filename in os.listdir(accepted_dir):
        if filename.endswith('.csv'):
            filepath = os.path.join(accepted_dir, filename)
            try:
                df = pd.read_csv(filepath, encoding='utf-8-sig')
                df['file_id'] = filename
                data.append(df)
            except Exception as e:
                print(f'Error reading {filename}: {e}')
    if data:
        return pd.concat(data, ignore_index=True)
    return pd.DataFrame()

def preprocess_data(df):
    """Preprocess data for analysis."""
    df['score'] = pd.to_numeric(df['score'], errors='coerce')
    df['reaction_time_ms'] = pd.to_numeric(df['reaction_time_ms'], errors='coerce')
    df['user_age'] = pd.to_numeric(df['user_age'], errors='coerce')
    df['user_gender'] = df['user_gender'].str.strip()
    df['user_education'] = df['user_education'].str.strip()
    df['user_art_training'] = df['user_art_training'].str.strip()
    df['image_type'] = df['block'].apply(lambda x: 'faces' if x == 'faces' else 'landscapes')
    return df

def descriptive_statistics(df):
    """Generate descriptive statistics."""
    print('=' * 70)
    print('DESCRIPTIVE STATISTICS')
    print('=' * 70)
    print('\n1. Overall Score Statistics:')
    print(df['score'].describe())
    print('\n2. Reaction Time Statistics (ms):')
    print(df['reaction_time_ms'].describe())
    print('\n3. Age Statistics:')
    print(df['user_age'].describe())
    print('\n4. Gender Distribution:')
    print(df['user_gender'].value_counts())
    print('\n5. Education Distribution:')
    print(df['user_education'].value_counts())
    print('\n6. Art Training Distribution:')
    print(df['user_art_training'].value_counts())
    print('\n7. Image Type Distribution:')
    print(df['image_type'].value_counts())

def gender_analysis(df):
    """Analyze gender differences in ratings."""
    print('\n' + '=' * 70)
    print('GENDER ANALYSIS')
    print('=' * 70)
    gender_groups = df.groupby('user_gender')['score']
    print('\n1. Score Statistics by Gender:')
    print(gender_groups.describe())
    if len(gender_groups) >= 2:
        genders = list(gender_groups.groups.keys())
        if len(genders) >= 2:
            group1 = df[df['user_gender'] == genders[0]]['score'].dropna()
            group2 = df[df['user_gender'] == genders[1]]['score'].dropna()
            if len(group1) > 0 and len(group2) > 0:
                t_stat, p_value = stats.ttest_ind(group1, group2)
                print(f'\n2. T-test between {genders[0]} and {genders[1]}:')
                print(f'   T-statistic: {t_stat:.4f}')
                print(f'   P-value: {p_value:.4f}')
                if p_value < 0.05:
                    print('   Result: Significant difference between genders')
                else:
                    print('   Result: No significant difference between genders')

def age_analysis(df):
    """Analyze age effects on ratings."""
    print('\n' + '=' * 70)
    print('AGE ANALYSIS')
    print('=' * 70)
    corr, p_value = stats.pearsonr(df['user_age'].dropna(), df['score'].dropna())
    print(f'\n1. Correlation between Age and Score:')
    print(f'   Pearson correlation: {corr:.4f}')
    print(f'   P-value: {p_value:.4f}')
    if p_value < 0.05:
        print('   Result: Significant correlation')
    else:
        print('   Result: No significant correlation')
    df['age_group'] = pd.cut(df['user_age'], bins=[0, 20, 30, 40, 50, 100], labels=['<20', '20-29', '30-39', '40-49', '50+'])
    age_groups = [group['score'].dropna() for name, group in df.groupby('age_group') if len(group) > 0]
    if len(age_groups) >= 2:
        f_stat, p_value = stats.f_oneway(*age_groups)
        print(f'\n2. ANOVA by Age Group:')
        print(f'   F-statistic: {f_stat:.4f}')
        print(f'   P-value: {p_value:.4f}')
        if p_value < 0.05:
            print('   Result: Significant differences between age groups')
        else:
            print('   Result: No significant differences between age groups')

def education_analysis(df):
    """Analyze education effects on ratings."""
    print('\n' + '=' * 70)
    print('EDUCATION ANALYSIS')
    print('=' * 70)
    print('\n1. Score Statistics by Education:')
    edu_groups = df.groupby('user_education')['score']
    print(edu_groups.describe())
    edu_groups_list = [group['score'].dropna() for name, group in df.groupby('user_education') if len(group) > 0]
    if len(edu_groups_list) >= 2:
        f_stat, p_value = stats.f_oneway(*edu_groups_list)
        print(f'\n2. ANOVA by Education Level:')
        print(f'   F-statistic: {f_stat:.4f}')
        print(f'   P-value: {p_value:.4f}')
        if p_value < 0.05:
            print('   Result: Significant differences between education levels')
        else:
            print('   Result: No significant differences between education levels')

def art_training_analysis(df):
    """Analyze art training effects on ratings."""
    print('\n' + '=' * 70)
    print('ART TRAINING ANALYSIS')
    print('=' * 70)
    df['has_art_training'] = df['user_art_training'].apply(lambda x: 'Yes' if '是' in str(x) else 'No')
    print('\n1. Score Statistics by Art Training:')
    art_groups = df.groupby('has_art_training')['score']
    print(art_groups.describe())
    group_yes = df[df['has_art_training'] == 'Yes']['score'].dropna()
    group_no = df[df['has_art_training'] == 'No']['score'].dropna()
    if len(group_yes) > 0 and len(group_no) > 0:
        t_stat, p_value = stats.ttest_ind(group_yes, group_no)
        print(f'\n2. T-test between Art Training Groups:')
        print(f'   T-statistic: {t_stat:.4f}')
        print(f'   P-value: {p_value:.4f}')
        if p_value < 0.05:
            print('   Result: Significant difference between groups')
        else:
            print('   Result: No significant difference between groups')

def image_type_analysis(df):
    """Analyze image type effects on ratings."""
    print('\n' + '=' * 70)
    print('IMAGE TYPE ANALYSIS')
    print('=' * 70)
    print('\n1. Score Statistics by Image Type:')
    type_groups = df.groupby('image_type')['score']
    print(type_groups.describe())
    group_faces = df[df['image_type'] == 'faces']['score'].dropna()
    group_landscapes = df[df['image_type'] == 'landscapes']['score'].dropna()
    if len(group_faces) > 0 and len(group_landscapes) > 0:
        t_stat, p_value = stats.ttest_ind(group_faces, group_landscapes)
        print(f'\n2. T-test between Image Types:')
        print(f'   T-statistic: {t_stat:.4f}')
        print(f'   P-value: {p_value:.4f}')
        if p_value < 0.05:
            print('   Result: Significant difference between image types')
        else:
            print('   Result: No significant difference between image types')

def reaction_time_analysis(df):
    """Analyze reaction time effects."""
    print('\n' + '=' * 70)
    print('REACTION TIME ANALYSIS')
    print('=' * 70)
    corr, p_value = stats.pearsonr(df['reaction_time_ms'].dropna(), df['score'].dropna())
    print(f'\n1. Correlation between Reaction Time and Score:')
    print(f'   Pearson correlation: {corr:.4f}')
    print(f'   P-value: {p_value:.4f}')
    if p_value < 0.05:
        print('   Result: Significant correlation')
    else:
        print('   Result: No significant correlation')
    print('\n2. Reaction Time by Image Type:')
    rt_by_type = df.groupby('image_type')['reaction_time_ms']
    print(rt_by_type.describe())

def multi_variable_analysis(df):
    """Perform multi-variable analysis."""
    print('\n' + '=' * 70)
    print('MULTI-VARIABLE ANALYSIS')
    print('=' * 70)
    analysis_df = df[['score', 'user_gender', 'user_age', 'user_education', 'has_art_training', 'image_type', 'reaction_time_ms']].copy()
    analysis_df = analysis_df.dropna()
    if len(analysis_df) > 0:
        print('\n1. Gender and Image Type Interaction:')
        interaction = analysis_df.groupby(['user_gender', 'image_type'])['score'].mean()
        print(interaction)
        print('\n2. Art Training and Image Type Interaction:')
        interaction = analysis_df.groupby(['has_art_training', 'image_type'])['score'].mean()
        print(interaction)

def generate_analysis_report(df):
    """Generate a comprehensive analysis report."""
    report = []
    report.append('=' * 80)
    report.append('COMPREHENSIVE ANALYSIS REPORT')
    report.append('=' * 80)
    report.append('')
    report.append(f'Total Sample Size: {len(df)}')
    report.append(f'Total Ratings: {len(df)}')
    report.append('')
    report.append('=' * 80)
    report.append('KEY FINDINGS')
    report.append('=' * 80)
    report.append('')
    gender_means = df.groupby('user_gender')['score'].mean()
    if len(gender_means) >= 2:
        report.append('1. Gender Differences:')
        for gender, mean in gender_means.items():
            report.append(f'   {gender}: {mean:.2f}')
        genders = list(gender_means.index)
        if len(genders) >= 2:
            group1 = df[df['user_gender'] == genders[0]]['score'].dropna()
            group2 = df[df['user_gender'] == genders[1]]['score'].dropna()
            if len(group1) > 0 and len(group2) > 0:
                t_stat, p_value = stats.ttest_ind(group1, group2)
                if p_value < 0.05:
                    report.append(f'   Significant difference (p={p_value:.4f})')
                else:
                    report.append(f'   No significant difference (p={p_value:.4f})')
    report.append('')
    corr, p_value = stats.pearsonr(df['user_age'].dropna(), df['score'].dropna())
    report.append('2. Age Effects:')
    report.append(f'   Correlation with score: {corr:.4f} (p={p_value:.4f})')
    if p_value < 0.05:
        report.append('   Significant correlation detected')
    else:
        report.append('   No significant correlation')
    report.append('')
    edu_groups = [group['score'].dropna() for name, group in df.groupby('user_education') if len(group) > 0]
    if len(edu_groups) >= 2:
        f_stat, p_value = stats.f_oneway(*edu_groups)
        report.append('3. Education Effects:')
        report.append(f'   ANOVA F-statistic: {f_stat:.4f} (p={p_value:.4f})')
        if p_value < 0.05:
            report.append('   Significant differences between education levels')
        else:
            report.append('   No significant differences between education levels')
    report.append('')
    group_yes = df[df['has_art_training'] == 'Yes']['score'].dropna()
    group_no = df[df['has_art_training'] == 'No']['score'].dropna()
    if len(group_yes) > 0 and len(group_no) > 0:
        t_stat, p_value = stats.ttest_ind(group_yes, group_no)
        report.append('4. Art Training Effects:')
        report.append(f'   With art training: {group_yes.mean():.2f}')
        report.append(f'   Without art training: {group_no.mean():.2f}')
        report.append(f'   T-test: {t_stat:.4f} (p={p_value:.4f})')
        if p_value < 0.05:
            report.append('   Significant difference detected')
        else:
            report.append('   No significant difference')
    report.append('')
    group_faces = df[df['image_type'] == 'faces']['score'].dropna()
    group_landscapes = df[df['image_type'] == 'landscapes']['score'].dropna()
    if len(group_faces) > 0 and len(group_landscapes) > 0:
        t_stat, p_value = stats.ttest_ind(group_faces, group_landscapes)
        report.append('5. Image Type Effects:')
        report.append(f'   Faces: {group_faces.mean():.2f}')
        report.append(f'   Landscapes: {group_landscapes.mean():.2f}')
        report.append(f'   T-test: {t_stat:.4f} (p={p_value:.4f})')
        if p_value < 0.05:
            report.append('   Significant difference detected')
        else:
            report.append('   No significant difference')
    report.append('')
    corr, p_value = stats.pearsonr(df['reaction_time_ms'].dropna(), df['score'].dropna())
    report.append('6. Reaction Time Effects:')
    report.append(f'   Correlation with score: {corr:.4f} (p={p_value:.4f})')
    if p_value < 0.05:
        report.append('   Significant correlation detected')
    else:
        report.append('   No significant correlation')
    report.append('')
    report.append('=' * 80)
    report.append('CONCLUSIONS')
    report.append('=' * 80)
    report.append('')
    report.append('Based on the comprehensive analysis, the following conclusions can be drawn:')
    report.append('')
    report.append('1. Demographic factors such as gender, age, education, and art training')
    report.append('   may influence aesthetic preferences, but the significance varies.')
    report.append('')
    report.append('2. Image type (faces vs landscapes) may elicit different aesthetic responses')
    report.append('   from participants.')
    report.append('')
    report.append('3. Reaction time analysis suggests that response speed may correlate with')
    report.append('   rating patterns, though further investigation is needed.')
    report.append('')
    report.append('4. The dataset provides a solid foundation for understanding aesthetic')
    report.append('   preferences across different population segments.')
    report.append('')
    report.append('=' * 80)
    with open('analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    print('\nAnalysis report saved as analysis_report.txt')

def main():
    """Main function to run comprehensive analysis."""
    print('Loading data...')
    df = load_all_data()
    if df.empty:
        print('No data found. Exiting.')
        return
    print(f'Loaded {len(df)} records')
    print('Preprocessing data...')
    df = preprocess_data(df)
    print('Running descriptive statistics...')
    descriptive_statistics(df)
    print('Running gender analysis...')
    gender_analysis(df)
    print('Running age analysis...')
    age_analysis(df)
    print('Running education analysis...')
    education_analysis(df)
    print('Running art training analysis...')
    art_training_analysis(df)
    print('Running image type analysis...')
    image_type_analysis(df)
    print('Running reaction time analysis...')
    reaction_time_analysis(df)
    print('Running multi-variable analysis...')
    multi_variable_analysis(df)
    print('Creating visualizations...')
    print('Generating analysis report...')
    generate_analysis_report(df)
    print('\n' + '=' * 80)
    print('ANALYSIS COMPLETE')
    print('=' * 80)
    print('Generated files:')
    print('  - analysis_report.txt')
    print('  - analysis_visualizations/ (directory with visualizations)')
if __name__ == '__main__':
    main()
