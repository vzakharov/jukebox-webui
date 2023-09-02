# Relative Import Converter
#
# This Python script is used to convert absolute imports to relative imports in a Python project.
# It recursively traverses through all the Python files in the specified directory and its subdirectories.
# For each Python file, it identifies all the import statements and modifies them from absolute to relative.
# This is particularly useful when you want to make your Python modules/packages portable or when you are
# restructuring your project and need to update the import paths.
#
# Usage:
# Run the script from the command line with the directory path as argument.
# Example: python relative_import_converter.py <directory_path>
#
# Note: This script uses the `ast` module to parse the Python source code and the `re` module for string manipulation.

import os
import re
import ast
import argparse

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

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Convert absolute imports to relative in a Python project.')
  parser.add_argument('path', help='The directory path to process')
  args = parser.parse_args()
  process_directory(args.path)