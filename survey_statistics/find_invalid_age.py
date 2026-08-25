"""
Find participants with invalid ages (under 14 or over 300)
and move them to Data/excluded directory.
"""

import os
import csv
import shutil
from collections import Counter

def find_invalid_age_files():
    """Find files with invalid ages."""
    accepted_dir = 'Data/accepted'
    files = [f for f in os.listdir(accepted_dir) if f.endswith('.csv')]
    
    invalid_files = []
    age_counter = Counter()
    
    for filename in files:
        filepath = os.path.join(accepted_dir, filename)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                age_str = row.get('user_age', '')
                try:
                    age = int(age_str)
                    age_counter[age] += 1
                    
                    # Check for invalid ages (under 15 or over 300)
                    if age < 15 or age > 300:
                        invalid_files.append({
                            'filename': filename,
                            'age': age,
                            'gender': row.get('user_gender', ''),
                            'reason': 'under 15' if age < 15 else 'over 300'
                        })
                        break  # Only count each file once
                except ValueError:
                    pass
                break  # Only read first row for user info
    
    return invalid_files, age_counter

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
            print(f"Moved: {file_info['filename']} (Age: {file_info['age']}, {file_info['reason']})")
    
    return moved_count

def main():
    """Main function."""
    print("Scanning for invalid age files...")
    invalid_files, age_counter = find_invalid_age_files()
    
    print(f"\nFound {len(invalid_files)} files with invalid ages:")
    
    under_15 = [f for f in invalid_files if f['reason'] == 'under 15']
    over_300 = [f for f in invalid_files if f['reason'] == 'over 300']
    
    print(f"  - Under 15 years old: {len(under_15)}")
    for f in under_15:
        print(f"    {f['filename']}: Age {f['age']}")
    
    print(f"  - Over 300 years old: {len(over_300)}")
    for f in over_300:
        print(f"    {f['filename']}: Age {f['age']}")
    
    # Show age distribution extremes
    print("\n" + "=" * 60)
    print("AGE DISTRIBUTION EXTREMES:")
    print("=" * 60)
    
    sorted_ages = sorted(age_counter.items())
    print(f"Youngest ages: {sorted_ages[:10]}")
    print(f"Oldest ages: {sorted_ages[-10:]}")
    
    if invalid_files:
        print("\n" + "=" * 60)
        print("MOVING FILES TO EXCLUDED DIRECTORY...")
        print("=" * 60)
        moved = move_files_to_excluded(invalid_files)
        print(f"\nSuccessfully moved {moved} files to Data/excluded/")
    else:
        print("\nNo invalid age files found.")

if __name__ == '__main__':
    main()
