import glob
import os
import subprocess
from datetime import datetime, timedelta

import librosa
import numpy as np
import yaml

from lib.audio.get_audio import get_audio
from lib.model.params import hps
from params import base_path

def reload_sample(project_name, sample_id, cut_out, last_n_sec, upsample_rendering, combine_levels, invert_center, filename, filename_without_hash):
  print('Nope, rendering...')

  # First, let's delete any old files that have the same name but a different hash (because these are now obsolete)
  for f in glob.glob(f'{filename_without_hash}.*'):
    print(f'(Deleting now-obsolete cached file {f})')
    os.remove(f)

  if last_n_sec:
    last_n_sec = -last_n_sec
  # (Because get_audio, called below, accepts "preview_sec", whose positive value means "preview the first n seconds", but we want to preview the last n seconds)

  wav, total_audio_length, upsampled_lengths = get_audio(project_name, sample_id, cut_out, last_n_sec, None, upsample_rendering, combine_levels, invert_center)
  # (level is None, which means "the highest level that is available", i.e. 2)

  if not os.path.exists(os.path.dirname(filename)):
    os.makedirs(os.path.dirname(filename))

  librosa.output.write_wav(f'{filename}.wav', np.asfortranarray(wav), hps.sr)

  # Add metadata (total audio length, upsampled lengths) to a yaml with the same name as the wav file
  with open(f'{filename}.yaml', 'w') as f:
    yaml.dump({
      'total_audio_length': total_audio_length,
      'upsampled_lengths': upsampled_lengths
    }, f)

  # Convert to mp3
  subprocess.run(['ffmpeg', '-y', '-i', f'{filename}.wav', '-acodec', 'libmp3lame', '-ab', '320k', f'{filename}.mp3'])

  # If there are more than 30 files in the rendered folder (i.e. more than 10 samples), delete the ones older than 1 day
  file_count_limit = 30
  files = glob.glob(f'{base_path}/{project_name}/rendered/*')
  if len(files) > file_count_limit:
    removed_count = 0
    failed_count = 0
    for f in files:
      try:
        if datetime.now() - datetime.fromtimestamp(os.path.getmtime(f)) > timedelta(days=1):
          os.remove(f)
          removed_count += 1
      except Exception as e:
        print(f'Could not remove {f}: {e}')
        failed_count += 1
    print(f'Deleted {removed_count} of {removed_count + failed_count} old files')
  
  return total_audio_length, upsampled_lengths