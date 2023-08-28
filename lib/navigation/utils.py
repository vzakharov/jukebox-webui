import os
import shutil
from datetime import datetime
from lib.navigation.get_children import get_children
from lib.navigation.get_parent import get_parent
from lib.navigation.get_projects import get_projects
from lib.upsampling.utils import get_first_upsampled_ancestor_zs, is_upsampled
import torch as t
from params import base_path


def get_zs_filename(project_name, sample_name, with_prefix = True):
  if not with_prefix:
    sample_name = f'{project_name}-{sample_name}'
  return f'{base_path}/{project_name}/{sample_name}.z'


def get_zs(project_name, sample_id, seek_upsampled = False):
  global base_path

  filename = f'{base_path}/{project_name}/{sample_id}.z'
  zs = t.load(filename)
  if not is_upsampled(zs) and seek_upsampled:
    upsampled_ancestor = get_first_upsampled_ancestor_zs(project_name, sample_id)
    if upsampled_ancestor:
      zs[:-1] = upsampled_ancestor[:-1]
  print(f'Loaded {filename}')
  return zs


def save_zs(zs, project_name, sample_id):
  global base_path

  filename = f'{base_path}/{project_name}/{sample_id}.z'
  t.save(zs, filename)
  print(f'Wrote {filename}')


def backup_sample(project_name, sample_id):
  global base_path

  current_filename = f'{base_path}/{project_name}/{sample_id}.z'
  timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
  backup_filename = f'{base_path}/{project_name}/bak/{sample_id}_{timestamp}.z'
  if not os.path.exists(os.path.dirname(backup_filename)):
    os.makedirs(os.path.dirname(backup_filename))
  # t.save(zs, backup_filename)
  shutil.copyfile(current_filename, backup_filename)
  print(f'Backed up {backup_filename}')


def get_prefix(project_name, parent_sample_id):
  return f'{parent_sample_id or project_name}-'


def get_project_name_from_sample_id(sample_id):
  projects = get_projects(include_new = False)
  # Find a project that matches the sample id, which is [project name]-[rest of sample id]
  for project_name in projects:
    if sample_id.startswith(f'{project_name}-'):
      return project_name


def get_siblings(project_name, sample_id):

  return get_children(project_name, get_parent(project_name, sample_id))


def is_new(project_name):
  return project_name == 'CREATE NEW' or not project_name


def is_ancestor(project_name, potential_ancestor, potential_descendant):
  parent = get_parent(project_name, potential_descendant)
  if parent == potential_ancestor:
    return True
  elif parent:
    return is_ancestor(project_name, potential_ancestor, parent)
  else:
    return False