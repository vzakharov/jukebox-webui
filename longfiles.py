import os
import sys

max_lines = int(sys.argv[1]) if len(sys.argv) > 1 else 50

def count_relevant_lines(filename):
  with open(filename, 'r') as file:
    lines = file.readlines()
  count = 0
  for line in lines:
    stripped = line.strip()
    if stripped:
      if not stripped.startswith('import') and not stripped.startswith('from') and not stripped.startswith('#'):
        count += 1
  return count

file_counts = {}

for root, dirs, files in os.walk('.'):
  for file in files:
    if file.endswith('.py'):
      filename = os.path.join(root, file)
      count = count_relevant_lines(filename)
      if count > max_lines:
        file_counts[filename] = count

# Sort the dictionary by its values in descending order
sorted_file_counts = sorted(file_counts.items(), key=lambda x: x[1], reverse=True)

# Print the filenames along with their counts
for filename, count in sorted_file_counts:
  print(f'{filename}: {count} lines')