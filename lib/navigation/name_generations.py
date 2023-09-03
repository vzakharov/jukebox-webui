import random

import torch as t

from .get_first_free_index import get_first_free_index
from .utils import get_prefix
from params import base_path

def name_generations(project_name, parent_sample_id, n_samples, zs):
  prefix = get_prefix(project_name, parent_sample_id)
  # For each sample, write the z (a subarray of zs)

  try:
    first_new_child_index = get_first_free_index(project_name, parent_sample_id)
  except Exception as e:
    print(f'Something went wrong: {e}')
    first_new_child_index = random.randrange(1e6, 1e7)
    print(f'Using random index {first_new_child_index} as a fallback')

  for i in range(n_samples):

    last_generated_id = f'{prefix}{first_new_child_index + i}'
    filename = f'{base_path}/{project_name}/{last_generated_id}'

    # zs is a list of 3 tensors, each of shape (n_samples, n_tokens)
    # To write the z for a single sample, we need to take a subarray of each tensor
    this_sample_zs = [ z[i:i+1] for z in zs ]

    t.save(this_sample_zs, f'{filename}.z')
    print(f'Wrote {filename}.z')

  return last_generated_id