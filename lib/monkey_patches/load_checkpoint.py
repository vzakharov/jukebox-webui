from lib.download_to_cache import download_to_cache
from params import models_path

import jukebox.utils.dist_adapter as dist
import torch as t
from jukebox.hparams import REMOTE_PREFIX

import os

def monkey_patched_load_checkpoint(path):
  global models_path
  restore = path
  if restore.startswith(REMOTE_PREFIX):
      remote_path = restore
      local_path = os.path.join(models_path, remote_path[len(REMOTE_PREFIX):])
      if dist.get_rank() % 8 == 0:
          download_to_cache(remote_path, local_path)
      restore = local_path
  dist.barrier()
  checkpoint = t.load(restore, map_location=t.device('cpu'))
  print("Restored from {}".format(restore))
  return checkpoint