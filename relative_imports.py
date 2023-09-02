import os
import re
import ast

def get_imports(source):
  tree = ast.parse(source)
  for node in tree.body:
    if isinstance(node, ast.Import):
      for alias in node.names:
        yield alias.name
    elif isinstance(node, ast.ImportFrom):
      yield node.module

def replace_imports(file_path, relative_path):
  with open(file_path, 'r') as file:
    source = file.read()
  imports = list(get_imports(source))
  for import_name in imports:
    if import_name.startswith(relative_path):
      source = re.sub(r'from ' + import_name + ' import', 'from .' + import_name[len(relative_path)+1:] + ' import', source)
  with open(file_path, 'w') as file:
    file.write(source)

def process_directory(path):
  for root, dirs, files in os.walk(path):
    for file in files:
      if file.endswith('.py'):
        relative_path = os.path.relpath(root, path).replace(os.sep, '.')
        replace_imports(os.path.join(root, file), relative_path)

process_directory('.')