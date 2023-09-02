import os
import subprocess
from .get_parent import get_parent
from .get_samples import get_samples
from .utils import get_zs_filename

import torch as t

from params import base_path

def prepare_purge_list(project_name):
  # Get the list of samples to purge
  # For each sample, get its parent sample
  # If the parent sample's level 0/1 is the same as this one, add it to the list
  samples = get_samples(project_name, False)
  purge_list = []
  for sample in samples:
    try:
      current_zs = t.load(get_zs_filename(project_name, sample))
      parent = get_parent(project_name, sample)
      if not parent:
        print(f'No parent for {sample}, skipping')
        continue
      parent_zs = t.load(get_zs_filename(project_name, parent))
      if (
            t.equal( current_zs[0], parent_zs[0] ) and
            t.equal( current_zs[1], parent_zs[1] ) and
            ( current_zs[0].shape[1] + current_zs[1].shape[1] > 0 )
          ):
        purge_list.append(sample)
        print(f'Adding {sample} to the purge list')
    except Exception as e:
      print(f'Error while processing {sample}: {e}, continuing')
      continue
  print(f'Purge list: {purge_list}')
  return '\n'.join(purge_list)

def purge_samples(project_name, purge_list):
  # Check the project size before purging, i.e. the joint size of all *.z files in the project directory
  project_size_before = subprocess.check_output(['du','-sh', os.path.join( base_path, project_name )]).split()[0].decode('utf-8')
  # For each sample in the purge list, delete it
  for sample in purge_list.split('\n'):
    zs = t.load(get_zs_filename(project_name, sample))
    level2_shape0 = zs[2].shape[0]
    purged_zs = [
          t.empty([level2_shape0,0], device='cuda'),
          t.empty([level2_shape0,0], device='cuda'),
          zs[2]
        ]
    print(f'Purged {sample} from shapes {[x.shape for x in zs]} to {[x.shape for x in purged_zs]}')
    t.save(purged_zs, get_zs_filename(project_name, sample))
  # Check the project size after purging
  project_size_after = subprocess.check_output(['du','-sh', os.path.join( base_path, project_name )]).split()[0].decode('utf-8')
  return f'Project size before: {project_size_before}, after: {project_size_after}'