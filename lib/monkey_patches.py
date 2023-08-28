from main import download_to_cache, models_path

import jukebox.utils.dist_adapter as dist
import torch as t
import librosa
from jukebox.hparams import REMOTE_PREFIX


import os

# Monkey patch load_audio, allowing for duration = None
def monkey_patched_load_audio(file, sr, offset, duration, mono=False):
  # Librosa loads more filetypes than soundfile
  x, _ = librosa.load(file, sr=sr, mono=mono, offset=offset/sr, duration=None if duration is None else duration/sr)
  if len(x.shape) == 1:
      x = x.reshape((1, -1))
  return x


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