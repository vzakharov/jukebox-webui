import os
import shutil
from datetime import datetime


from params import base_path


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

def is_new(project_name):
  return project_name == 'CREATE NEW' or not project_name

loaded_settings = {}