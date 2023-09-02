import torch as t

from lib.upsampling.utils import get_first_upsampled_ancestor_zs, is_upsampled
from params import base_path


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


def get_zs_filename(project_name, sample_name, with_prefix = True):
  if not with_prefix:
    sample_name = f'{project_name}-{sample_name}'
  return f'{base_path}/{project_name}/{sample_name}.z'