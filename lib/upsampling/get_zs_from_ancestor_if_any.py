from .Upsampling import Upsampling
from .utils import get_first_upsampled_ancestor_zs

import torch as t

def get_zs_from_ancestor_if_any(project_name, sample_id):
    for i in range( len(Upsampling.zs) ):
      if i == 2:
        Upsampling.zs[i] = Upsampling.zs[i][0].repeat(3, 1)
      elif Upsampling.zs[i].shape[0] != 3:
      # If there are no upsampled ancestors, replace with an empty tensor
        first_upsampled_ancestor = get_first_upsampled_ancestor_zs(project_name, sample_id)
        if not first_upsampled_ancestor:
          print(f'No upsampled ancestors found for {sample_id}, using empty tensors')
          Upsampling.zs[i] = t.empty( (3, 0), dtype=t.int64 ).cuda()
        else:
          print(f'Using first upsampled ancestor zs for {sample_id}')
          Upsampling.zs[i] = first_upsampled_ancestor[i]