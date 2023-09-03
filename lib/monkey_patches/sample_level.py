import os
import signal
from datetime import datetime

from jukebox.utils.sample_utils import get_starts

from lib.upsampling.Upsampling import Upsampling
from params import base_path

from .upsample_window import upsample_window


def monkey_patched_sample_level(zs, labels, sampling_kwargs, level, prior, total_length, hop_length, hps):

  global base_path

  # The original code provides for shorter samples by sampling only a partial window, but we'll just throw an error for simplicity
  assert total_length >= prior.n_ctx, f'Total length {total_length} is shorter than prior.n_ctx {prior.n_ctx}'

  Upsampling.zs = zs
  Upsampling.level = level
  Upsampling.prior = prior
  Upsampling.labels = labels
  Upsampling.hps = hps
  Upsampling.kwargs = sampling_kwargs

  print(f"Sampling level {level}")
  # Remember current time
  start_time = datetime.now()
  Upsampling.windows = get_starts(total_length, prior.n_ctx, hop_length)

  print(f'Totally {len(Upsampling.windows)} windows at level {level}')

  # Remove all windows whose start + n_ctx is less than however many samples we've already upsampled (at this level)
  already_upsampled = Upsampling.zs[level].shape[1]
  if already_upsampled > 0:
    print(f'Already upsampled {already_upsampled} samples at level {level}')
    Upsampling.windows = [ start for start in Upsampling.windows if start + prior.n_ctx > already_upsampled ]

  if len(Upsampling.windows) == 0:
    print(f'No windows to upsample at level {Upsampling.level}')
  else:
    print(f'Upsampling {len(Upsampling.windows)} windows, from {Upsampling.windows[0]} to {Upsampling.windows[-1]+prior.n_ctx}')

    Upsampling.window_index = 0
    for start in Upsampling.windows:

      if upsample_window(start) == 'break':
        break

  if level == 0:
    Upsampling.running = False
    if Upsampling.kill_runtime_once_done:
      print('Killing process')
      os.kill(os.getpid(), signal.SIGKILL)

  return Upsampling.zs

