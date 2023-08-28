from lib.audio.get_audio import get_audio
from lib.navigation.get_children import get_children
from lib.navigation.get_parent import get_parent
from lib.navigation.get_sample_filename import get_sample_filename
from lib.ui.UI import UI
from lib.model.params import hps
from params import base_path


import gradio as gr
import librosa
import numpy as np
import yaml


import glob
import hashlib
import os
import random
import subprocess
from datetime import datetime, timedelta


def get_sample(project_name, sample_id, cut_out='', last_n_sec=None, upsample_rendering=4, combine_levels=True, invert_center=False, force_reload=False):

  global hps

  print(f'Loading sample {sample_id}')

  filename = get_sample_filename(project_name, sample_id, cut_out, last_n_sec, upsample_rendering, combine_levels, invert_center)
  filename_without_hash = filename

  # Add a hash (8 characters of md5) of the corresponding z file (so that we can detect if the z file has changed and hence we need to re-render)
  filename += f'{hashlib.md5(open(f"{base_path}/{project_name}/{sample_id}.z", "rb").read()).hexdigest()[:8]} '


  print(f'Checking if {filename} is cached...')

  # Make sure all 3 of wav, mp3 and yaml exist
  if not force_reload:
    for ext in [ 'wav', 'mp3', 'yaml' ]:
      if not os.path.isfile(f'{filename}.{ext}'):
        force_reload = True
        break

  if force_reload:

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

  else:
    print('Yep, using it.')
    wav = None

    # Load metadata
    with open(f'{filename}.yaml', 'r') as f:
      metadata = yaml.load(f, Loader=yaml.FullLoader)
      total_audio_length = metadata['total_audio_length']
      upsampled_lengths = metadata['upsampled_lengths']
      print(f'(Also loaded metadata: {metadata})')

  chunk_filenames = []

  # If the mp3 size is > certain sie, we'll need to send it back in chunks, so we divide the mp3 into as many chunks as needed
  file_size = os.path.getsize(f'{filename}.mp3')
  file_limit = 300000

  if file_size > file_limit:
    print(f'MP3 file size is {file_size} bytes, splitting into chunks...')
    file_content = open(f'{filename}.mp3', 'rb').read()
    for i in range(0, file_size, file_limit):
      # Place the chunk file in tmp/[filename without path] [range].mp3_chunk
      # Create the tmp folder if it doesn't exist
      if not os.path.exists(f'tmp'):
        os.makedirs(f'tmp')
      chunk_filename = f'tmp/{os.path.basename(filename)}{i}-{i+file_limit} .mp3_chunk'
      with open(chunk_filename, 'wb') as f:
        f.write(file_content[i:i+file_limit])
        print(f'Wrote bytes {i}-{i+file_limit} to {chunk_filename}')
      chunk_filenames.append(chunk_filename)
  else:
    chunk_filenames = [f'{filename}.mp3']

  print(f'Files to send: {chunk_filenames}')

  return {
    UI.current_chunks: chunk_filenames,
    UI.total_audio_length: total_audio_length,
    UI.go_to_children_button: gr.update(
      visible = len(get_children(project_name, sample_id)) > 0
    ),
    UI.go_to_parent_button: gr.update(
      visible = get_parent(project_name, sample_id) is not None
    ),
    UI.sample_box: gr.update(
      visible = True
    ),
    UI.upsampled_lengths: ','.join([str(length) for length in upsampled_lengths]),
    # Random number for picked sample updated flag
    UI.picked_sample_updated: random.random(),
  }