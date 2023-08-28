from lib.navigation.get_custom_parents import get_custom_parents
from params import base_path


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