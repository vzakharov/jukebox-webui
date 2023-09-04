from lib.utils import seconds_to_tokens
from params import base_path

import torch as t

def prepare_zs(project_name, parent_sample_id, n_samples, discard_window):
  discarded_zs = None
  if parent_sample_id:
    zs = t.load(f'{base_path}/{project_name}/{parent_sample_id}.z')
    print(f'Loaded parent sample {parent_sample_id} of shape {[ z.shape for z in zs ]}')
    zs = [ z[0].repeat(n_samples, 1) for z in zs ]
    print(f'Converted to shape {[ z.shape for z in zs ]}')

    if discard_window > 0:
      discarded_zs = [ z[:, :seconds_to_tokens(discard_window)] for z in zs ]
      zs = [ z[:, seconds_to_tokens(discard_window):] for z in zs ]
      print(f'Discarded the first {discard_window} seconds, now zs are of shape {[ z.shape for z in zs ]}')

  else:
    zs = [ t.zeros(n_samples, 0, dtype=t.long, device='cuda') for _ in range(3) ]
    print('No parent sample or primer provided, starting from scratch')
  return zs, discarded_zs