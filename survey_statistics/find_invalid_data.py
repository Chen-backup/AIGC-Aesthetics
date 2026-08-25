"""
Find participants with invalid data (invalid ages or other gender)
and move them to Data/excluded directory.
"""
import os
import csv
import shutil
from collections import Counter

def find_invalid_files():
    """Find files with invalid ages or other gender."""
    accepted_dir = 'Data/accepted'
    files = [f for f in os.listdir(accepted_dir) if f.endswith('.csv')]
    invalid_files = []
    age_counter = Counter()
    gender_counter = Counter()
    for filename in files:
        filepath = os.path.join(accepted_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                age_str = row.get('user_age', '')
                gender = row.get('user_gender', '')
                gender_counter[gender] += 1
                if gender == '其他':
                    invalid_files.append({'filename': filename, 'age': age_str, 'gender': gender, 'reason': 'other gender'})
                    break
                try:
                    age = int(age_str)
                    age_counter[age] += 1
                    if age < 15 or age > 300:
                        invalid_files.append({'filename': filename, 'age': age, 'gender': gender, 'reason': 'under 15' if age < 15 else 'over 300'})
                        break
                except ValueError:
                    pass
                break
    return (invalid_files, age_counter, gender_counter)

def move_files_to_excluded(invalid_files):
    """Move invalid files to excluded directory."""
    accepted_dir = 'Data/accepted'
    excluded_dir = 'Data/excluded'
    os.makedirs(excluded_dir, exist_ok=True)
    moved_count = 0
    for file_info in invalid_files:
        src = os.path.join(accepted_dir, file_info['filename'])
        dst = os.path.join(excluded_dir, file_info['filename'])
        if os.path.exists(src):
            shutil.move(src, dst)
            moved_count += 1
            if file_info['reason'] == 'other gender':
                print(f'Moved: {file_info['filename']} (Gender: {file_info['gender']}, {file_info['reason']})')
            else:
                print(f'Moved: {file_info['filename']} (Age: {file_info['age']}, {file_info['reason']})')
    return moved_count

def main():
    """Main function."""
    print('Scanning for invalid data files...')
    invalid_files, age_counter, gender_counter = find_invalid_files()
    print(f'\nFound {len(invalid_files)} files with invalid data:')
    under_15 = [f for f in invalid_files if f['reason'] == 'under 15']
    over_300 = [f for f in invalid_files if f['reason'] == 'over 300']
    other_gender = [f for f in invalid_files if f['reason'] == 'other gender']
    print(f'  - Under 15 years old: {len(under_15)}')
    for f in under_15:
        print(f'    {f['filename']}: Age {f['age']}')
    print(f'  - Over 300 years old: {len(over_300)}')
    for f in over_300:
        print(f'    {f['filename']}: Age {f['age']}')
    print(f'  - Other gender: {len(other_gender)}')
    for f in other_gender:
        print(f'    {f['filename']}: Gender {f['gender']}')
    print('\n' + '=' * 60)
    print('GENDER DISTRIBUTION:')
    print('=' * 60)
    for gender, count in sorted(gender_counter.items()):
        print(f'  {gender}: {count}')
    if age_counter:
        print('\n' + '=' * 60)
        print('AGE DISTRIBUTION EXTREMES:')
        print('=' * 60)
        sorted_ages = sorted(age_counter.items())
        print(f'Youngest ages: {sorted_ages[:10]}')
        print(f'Oldest ages: {sorted_ages[-10:]}')
    if invalid_files:
        print('\n' + '=' * 60)
        print('MOVING FILES TO EXCLUDED DIRECTORY...')
        print('=' * 60)
        moved = move_files_to_excluded(invalid_files)
        print(f'\nSuccessfully moved {moved} files to Data/excluded/')
    else:
        print('\nNo invalid data files found.')
if __name__ == '__main__':
    main()
