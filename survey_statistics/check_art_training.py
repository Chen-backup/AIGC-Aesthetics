import os
import csv
from collections import Counter

# Count art training from Data/accepted directory
accepted_dir = 'Data/accepted'
files = [f for f in os.listdir(accepted_dir) if f.endswith('.csv')]

art_training_counter = Counter()
for filename in files:
    filepath = os.path.join(accepted_dir, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            art_training = row.get('user_art_training', '')
            art_training_counter[art_training] += 1
            break  # Only read first row

print("Art training distribution:")
for training, count in sorted(art_training_counter.items()):
    print(f"  {training}: {count}")
print(f"Total: {sum(art_training_counter.values())}")
