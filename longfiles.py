import os

def count_relevant_lines(filename):
  with open(filename, 'r') as file:
    lines = file.readlines()
  count = 0
  for line in lines:
    stripped = line.strip()
    if not stripped.startswith('import') and not stripped.startswith('from') and not stripped.startswith('#'):
      count += 1
  return count

for root, dirs, files in os.walk('.'):
  for file in files:
    if file.endswith('.py'):
      filename = os.path.join(root, file)
      if count_relevant_lines(filename) > 50:
        print(filename)