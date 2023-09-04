from datetime import datetime

import gradio as gr
import torch as t

from lib.model.model import Model
from lib.model.params import hps
from lib.navigation.get_first_free_index import get_first_free_index
from lib.navigation.get_samples import get_samples
from lib.ui.elements.first import first_generation_row, prime_timestamp
from lib.ui.elements.navigation import sample_tree
from lib.ui.utils import HIDE
from params import base_path

from .to_x import to_x


def to_sample(project_name, audio, sec_to_trim_primed_audio, show_leafs_only):

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

  x = to_x(audio)

  zs = Model.top_prior.encode( x, start_level=0, end_level=len(Model.priors), bs_chunks=x.shape[0] )
  print(f'Encoded audio to zs of shape {[ z.shape for z in zs ]}')

  primed_sample_id = f'{project_name}-{get_first_free_index(project_name)}'
  filename = f'{base_path}/{project_name}/{primed_sample_id}.z'
  t.save(zs, filename)

  return {
    sample_tree: gr.update(
      choices = get_samples(project_name, show_leafs_only),
      value = primed_sample_id
    ),
    prime_timestamp: datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
    first_generation_row: HIDE,
  }