from lib.model.params import hps
from lib.navigation.get_samples import get_samples
from lib.navigation.get_first_free_index import get_first_free_index
from lib.ui.UI import UI
from lib.ui.utils import HIDE
from main import device, priors, top_prior
from params import base_path

import gradio as gr
import librosa
import numpy as np
import torch as t

from datetime import datetime

def convert_audio_to_sample(project_name, audio, sec_to_trim_primed_audio, show_leafs_only):

  global hps, base_path

  # audio is of the following form:
  # (48000, array([        0,         0,         0, ..., -26209718, -25768554,       -25400996], dtype=int32))
  # dtype=int16 is also possible
  # Or, if it is a stereo file, audio[1] is [[left, right], [left, right], ...]

  print(f'Audio: {audio}')
  # breakpoint()
  print(f'Audio length: {len(audio[1])} samples, sample rate: {audio[0]} Hz')

  # If it is a stereo file, we need to convert it to mono by averaging the left and right channels
  if len(audio[1].shape) > 1:
    audio = (audio[0], audio[1].mean(axis=1))
    print(f'Converted stereo to mono (shape: {audio[1].shape})')

  if sec_to_trim_primed_audio:
    audio = (audio[0], audio[1][:int(audio[0] * sec_to_trim_primed_audio)])
    print(f'Trimmed audio to {sec_to_trim_primed_audio} seconds')

  # Convert the audio to float depending on the dtype

  x = audio[1] / 2**31 if audio[1].dtype == np.int32 else audio[1] / 2**15
  print(f'Converted to [-1, 1]; min = {x.min()}, max = {x.max()}')

  # Resample the audio to hps.sr (if needed)

  if audio[0] != hps.sr:
    x = librosa.resample(x, audio[0], hps.sr)
    print(f'Resampled audio to {hps.sr}')

  # Convert the audio to a tensor (e.g. from array([[-1.407e-03, -4.461e-04, ..., -3.042e-05,  1.277e-05]], dtype=float32) to tensor([[-1.407e-03], [-4.461e-04], ..., [-3.042e-05], [ 1.277e-05]], dtype=float32))

  if len(x.shape) == 1:
    x = x.reshape((1, -1))
    print(f'Reshaped audio to {x.shape}')

  x = x.T
  print(f'Transposed audio to {x.shape}')

  xs = [ x ]

  print(f'Created {len(xs)} samples of {x.shape} shape each')

  x = t.stack([t.from_numpy(x) for x in xs])
  print(f'Stacked samples to {x.shape}')

  x = x.to(device, non_blocking=True)
  print(f'Moved samples to {device}')

  zs = top_prior.encode( x, start_level=0, end_level=len(priors), bs_chunks=x.shape[0] )
  print(f'Encoded audio to zs of shape {[ z.shape for z in zs ]}')

  primed_sample_id = f'{project_name}-{get_first_free_index(project_name)}'
  filename = f'{base_path}/{project_name}/{primed_sample_id}.z'
  t.save(zs, filename)

  return {
    UI.sample_tree: gr.update(
      choices = get_samples(project_name, show_leafs_only),
      value = primed_sample_id
    ),
    UI.prime_timestamp: datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
    UI.first_generation_row: HIDE,
  }