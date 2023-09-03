import os
import re

from params import base_path

from .get_custom_parents import get_custom_parents
from .utils import get_prefix


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


def get_parent(project_name, sample_id):

  global base_path

  custom_parents = get_custom_parents(project_name)

  if sample_id in custom_parents:
    return custom_parents[sample_id]

  # Remove the project name and first dash from the sample id
  path = sample_id[ len(project_name) + 1: ].split('-')
  parent_sample_id = '-'.join([ project_name, *path[:-1] ]) if len(path) > 1 else None
  # print(f'Parent of {sample_id}: {parent_sample_id}')
  return parent_sample_id


def get_siblings(project_name, sample_id):
  return get_children(project_name, get_parent(project_name, sample_id))


def is_ancestor(project_name, potential_ancestor, potential_descendant):
  parent = get_parent(project_name, potential_descendant)
  if parent == potential_ancestor:
    return True
  elif parent:
    return is_ancestor(project_name, potential_ancestor, parent)
  else:
    return False