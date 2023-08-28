from lib.navigation.get_custom_parents import get_custom_parents
from lib.navigation.utils import get_prefix
from params import base_path


import os
import re


def get_children(project_name, parent_sample_id, include_custom=True):

  global base_path

  prefix = get_prefix(project_name, parent_sample_id)
  child_ids = []
  for filename in os.listdir(f'{base_path}/{project_name}'):
    match = re.match(f'{prefix}(\d+)\\.zs?$', filename)
    if match:
      child_ids += [ filename.split('.')[0] ]

  if include_custom:

    custom_parents = get_custom_parents(project_name)

    for sample_id in custom_parents:
      if custom_parents[sample_id] == parent_sample_id:
        child_ids += [ sample_id ]

  # print(f'Children of {parent_sample_id}: {child_ids}')

  # Sort alphabetically
  child_ids.sort()

  return child_ids