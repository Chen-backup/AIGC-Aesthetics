import os
import csv
from collections import Counter
accepted_dir = 'Data/accepted'
files = [f for f in os.listdir(accepted_dir) if f.endswith('.csv')]
education_counter = Counter()
total_files = len(files)
education_available = 0
for filename in files:
    filepath = os.path.join(accepted_dir, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        first_row = next(reader, None)
        if first_row and 'user_education' in first_row:
            education = first_row['user_education']
            if education:
                education_counter[education] += 1
                education_available += 1
print(f'Total users: {total_files}')
print(f'Users with education data: {education_available}')
print(f'Users missing education data: {total_files - education_available}')
print(f'Missing rate: {(total_files - education_available) / total_files * 100:.1f}%')
print('\nEducation distribution:')
for edu, count in sorted(education_counter.items()):
    print(f'{edu}: {count}')
