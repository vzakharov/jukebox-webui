import hashlib
import os

import yaml

from lib.model.params import hps
from params import base_path

from .get_sample_filename import get_sample_filename
from .get_sample_return_data import get_sample_return_data
from .reload_sample import reload_sample
from .split_into_chunks import split_into_chunks


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

    total_audio_length, upsampled_lengths = reload_sample(
      project_name, sample_id, cut_out, last_n_sec, upsample_rendering, combine_levels, invert_center, filename, filename_without_hash
    )

  else:

    print('Yep, using it.')
    wav = None

    # Load metadata
    with open(f'{filename}.yaml', 'r') as f:
      metadata = yaml.load(f, Loader=yaml.FullLoader)
      total_audio_length = metadata['total_audio_length']
      upsampled_lengths = metadata['upsampled_lengths']
      print(f'(Also loaded metadata: {metadata})')

  chunk_filenames = split_into_chunks(filename)

  return get_sample_return_data(project_name, sample_id, total_audio_length, upsampled_lengths, chunk_filenames)