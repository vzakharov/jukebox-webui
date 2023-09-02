import os
import re
import sys

def rewrite_file(filename, module_regex):
  print(f'Rewriting {filename}...')
  with open(filename, 'r') as file:
    source = file.read()

  imports = re.findall(r'(?<=import )' + module_regex + r'$', source, re.MULTILINE)
  print(f'Found imports: {imports}')

  for module in imports:
    print(f'Processing {module}...')
    usages = re.findall(module + r'\.(\w+)', source)
    print(f'Found usages: {usages}')

    source = re.sub(r'import ' + module + r'(?!\s+as)', 'from ' + module + ' import ' + ', '.join(usages), source)

    for usage in usages:
      source = source.replace(module + '.' + usage, usage)

  with open(filename, 'w') as file:
    file.write(source)

def rewrite_all_files(directory, module_regex):
  for root, dirs, files in os.walk(directory):
    for file in files:
      if file.endswith('.py'):
        rewrite_file(os.path.join(root, file), module_regex)

rewrite_all_files(sys.argv[1], sys.argv[2])