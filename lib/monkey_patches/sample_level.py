from lib.utils import as_local_hh_mm
from lib.ui.UI import UI
from lib.upsampling.Upsampling import Upsampling
from main import sample_id_to_restart_upsampling_with


import torch as t
from jukebox.sample import sample_single_window
from jukebox.utils.sample_utils import get_starts


from datetime import datetime

from lib.upsampling.restart_upsampling import restart_upsampling
from params import base_path

def monkey_patched_sample_level(zs, labels, sampling_kwargs, level, prior, total_length, hop_length, hps):

  global base_path

  # The original code provides for shorter samples by sampling only a partial window, but we'll just throw an error for simplicity
  assert total_length >= prior.n_ctx, f'Total length {total_length} is shorter than prior.n_ctx {prior.n_ctx}'

  Upsampling.zs = zs
  Upsampling.level = level

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
    print(f'No windows to upsample at level {level}')
  else:
    print(f'Upsampling {len(Upsampling.windows)} windows, from {Upsampling.windows[0]} to {Upsampling.windows[-1]+prior.n_ctx}')

    Upsampling.window_index = 0
    for start in Upsampling.windows:

      if Upsampling.stop:
        print(f'Upsampling stopped for level {level}')
        if Upsampling.level == 0:
          Upsampling.stop = False
        Upsampling.running = False

        if sample_id_to_restart_upsampling_with is not None:
          print(f'Upsampling will be restarted for sample {sample_id_to_restart_upsampling_with}')
          restart_upsampling(sample_id_to_restart_upsampling_with)

        break

      Upsampling.window_start_time = datetime.now()
      Upsampling.windows_remaining = len(Upsampling.windows) - Upsampling.window_index
      Upsampling.time_remaining = Upsampling.time_per_window * Upsampling.windows_remaining
      Upsampling.eta = datetime.now() + Upsampling.time_remaining

      Upsampling.status_markdown = f'Upsampling **window { Upsampling.window_index+1 } of { len(Upsampling.windows) }** for the **{ UI.UPSAMPLING_LEVEL_NAMES[2-level] }** level.\n\nEstimated level completion: **{ as_local_hh_mm(Upsampling.eta) }** your time.'

      # Print the status with an hourglass emoji in front of it
      print(f'\n\n⏳ {Upsampling.status_markdown}\n\n')

      Upsampling.zs = sample_single_window(Upsampling.zs, labels, sampling_kwargs, level, prior, start, hps)

      # Only update time_per_window we've sampled at least 2 windows (as the first window can take either a long or short time due to its size)
      if Upsampling.window_index > 1:
        Upsampling.time_per_window = datetime.now() - Upsampling.window_start_time

      path = f'{base_path}/{Upsampling.project}/{Upsampling.sample_id}.z'
      print(f'Saving upsampled z to {path}')
      t.save(Upsampling.zs, path)
      print('Done.')
      Upsampling.should_refresh_audio = True
      Upsampling.window_index += 1

  if level == 0:
    Upsampling.running = False
    if Upsampling.kill_runtime_once_done:
      print('Killing runtime')
      runtime.unassign()

  return Upsampling.zs