from datetime import datetime

import torch as t
from jukebox.sample import sample_single_window

from lib.ui.elements.upsampling import level_names
from lib.upsampling.Upsampling import Upsampling
from lib.upsampling.restart_upsampling import restart_upsampling
from lib.utils import as_local_hh_mm
from main import sample_id_to_restart_upsampling_with
from params import base_path

def upsample_window(start):

  if Upsampling.stop:

    print(f'Upsampling stopped for level {Upsampling.level}')
    if Upsampling.level == 0:
      Upsampling.stop = False
    Upsampling.running = False

    if sample_id_to_restart_upsampling_with is not None:
      print(f'Upsampling will be restarted for sample {sample_id_to_restart_upsampling_with}')
      restart_upsampling(sample_id_to_restart_upsampling_with)

    return 'break'
  
  Upsampling.window_start_time = datetime.now()
  Upsampling.windows_remaining = len(Upsampling.windows) - Upsampling.window_index
  Upsampling.time_remaining = Upsampling.time_per_window * Upsampling.windows_remaining
  Upsampling.eta = datetime.now() + Upsampling.time_remaining

  Upsampling.status_markdown = f'Upsampling **window { Upsampling.window_index+1 } of { len(Upsampling.windows) }** for the **{ level_names[2-Upsampling.level] }** level.\n\nEstimated level completion: **{ as_local_hh_mm(Upsampling.eta) }** your time.'

    # Print the status with an hourglass emoji in front of it
  print(f'\n\n⏳ {Upsampling.status_markdown}\n\n')

  Upsampling.zs = sample_single_window(Upsampling.zs, Upsampling.labels, Upsampling.kwargs, Upsampling.level, Upsampling.prior, start, Upsampling.hps)

    # Only update time_per_window we've sampled at least 2 windows (as the first window can take either a long or short time due to its size)
  if Upsampling.window_index > 1:
    Upsampling.time_per_window = datetime.now() - Upsampling.window_start_time

  path = f'{base_path}/{Upsampling.project}/{Upsampling.sample_id}.z'
  print(f'Saving upsampled z to {path}')
  t.save(Upsampling.zs, path)
  print('Done.')
  Upsampling.should_refresh_audio = True
  Upsampling.window_index += 1