import os
import shutil
from datetime import datetime

import gradio as gr

from params import base_path

from .get_children import get_children
from .get_parent import get_parent
from .get_projects import get_projects


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

loaded_settings = {}
input_names = { input: name for name, input in locals().items() if isinstance(input, gr.components.FormComponent) }
inputs_by_name = { name: input for name, input in locals().items() if isinstance(input, gr.components.FormComponent) }