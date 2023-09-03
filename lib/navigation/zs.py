import torch as t

from params import base_path


def get_zs(project_name, sample_id):
  global base_path

  filename = f'{base_path}/{project_name}/{sample_id}.z'
  zs = t.load(filename)
  print(f'Loaded {filename}')
  return zs

def save_zs(zs, project_name, sample_id):
  global base_path

  filename = f'{base_path}/{project_name}/{sample_id}.z'
  t.save(zs, filename)
  print(f'Wrote {filename}')


def get_zs_filename(project_name, sample_name, with_prefix = True):
  if not with_prefix:
    sample_name = f'{project_name}-{sample_name}'
  return f'{base_path}/{project_name}/{sample_name}.z'