import subprocess

import gradio as gr

from UI.project import max_n_samples, max_n_samples, n_samples, total_audio_length, total_audio_length, max_n_samples
from UI.preview import 

from lib.app import app

def set_max_n_samples( total_audio_length ):

  max_n_samples_by_gpu_and_duration = {
    'Tesla T4': {
      0: 4,
      8.5: 3,
      13: 2
    }
    # The key indicates the audio length threshold in seconds trespassing which makes max_n_samples equal to the value
  }

  # Get GPU via nvidia-smi
  gpu = subprocess.check_output( 'nvidia-smi --query-gpu=gpu_name --format=csv,noheader', shell=True ).decode('utf-8').strip()

  # The default value is 4
  max_n_samples = 4
  if gpu in max_n_samples_by_gpu_and_duration and total_audio_length:
    # Get the max n samples for the GPU from the respective dict
    max_n_samples_for_gpu = max_n_samples_by_gpu_and_duration[gpu]
    max_n_samples_above_threshold = [ max_n_samples_for_gpu[threshold] for threshold in max_n_samples_for_gpu if total_audio_length > threshold ]
    if len(max_n_samples_above_threshold) > 0:
      max_n_samples = min( max_n_samples_for_gpu[ threshold ] for threshold in max_n_samples_for_gpu if total_audio_length > threshold )

  return max_n_samples

def adjust_max_samples():
  max_n_samples.render().change(
    inputs = max_n_samples,
    outputs = n_samples,
    fn = lambda max_n_samples: gr.update(
      maximum = max_n_samples,
      # value = min( n_samples, max_n_samples ),
    )
  )

  # Do this on audio length change and app load
  for handler in [ total_audio_length.change, app.load ]:
    handler(
      inputs = total_audio_length,
      outputs = max_n_samples,
      fn = set_max_n_samples,
    )