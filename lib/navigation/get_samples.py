from lib.navigation.get_children import get_children
from params import base_path

import os
import re

def get_samples(project_name, leafs_only):

  choices = []
  for filename in os.listdir(f'{base_path}/{project_name}'):
    if re.match(r'.*\.zs?$', filename):
      id = filename.split('.')[0]
      if leafs_only and len( get_children(project_name, id) ) > 0:
        continue
      choices += [ id ]

  # Sort by id, in descending order
  choices.sort(reverse = True)

  return choices