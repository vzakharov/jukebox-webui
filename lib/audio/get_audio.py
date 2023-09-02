from .decode_short import decode_short
from .decode import decode
from .to_stereo import to_stereo
from .cut import cut_z
from lib.model.params import hps
from lib.navigation.utils import get_zs
from lib.upsampling.utils import get_levels
from lib.utils import seconds_to_tokens, tokens_to_seconds
from params import base_path

import numpy as np
import torch as t
from jukebox.utils.torch_utils import empty_cache

def get_audio(project_name, sample_id, cut_audio, preview_sec, level=None, stereo_rendering=3, combine_levels=True, invert_center=False):

  print(f'Generating audio for {project_name}/{sample_id} (level {level}, stereo rendering {stereo_rendering}, combine levels {combine_levels})')
  print(f'Cut: {cut_audio}, preview: {preview_sec}')

  # Get current GPU memory usage. If it's above 12GB, empty the cache
  memory = t.cuda.memory_allocated()
  print(f'GPU memory usage is {memory / 1e9:.1f} GB')
  if t.cuda.memory_allocated() > 12e9:
    print('GPU memory usage is above 12GB, clearing the cache')
    empty_cache()
    print(f'GPU memory usage is now {t.cuda.memory_allocated() / 1e9:1f} GB')

  global base_path, hps

  zs = get_zs(project_name, sample_id, seek_upsampled=True)

  # If no level is specified, use 2 (and then go downwards if combine_levels is True)
  if level is None:
    level = 2

  z = zs[level]
  # print(f'Loaded {filename}.z at level {level}, shape: {z.shape}')

  if cut_audio:
    z = cut_z(z, cut_audio, level)

  audio_length = int( tokens_to_seconds(z.shape[1], level) * 100 ) / 100

  if preview_sec:
    seconds_to_cut_from_start = audio_length - abs(preview_sec) if preview_sec < 0 else preview_sec
    z = cut_z(z, f'-{seconds_to_cut_from_start}', level)
  else:
    seconds_to_cut_from_start = 0

  wav = decode(z, level)

  # wav is now of shape (n_samples, sample_length, 1)
  # If this is level 2, we want just (sample_length,), picking the first sample if there are multiple
  if level == 2:
    wav = wav[0, :, 0]

  # Otherwise, this is a batch of upsampled audio, so we need to act depending on the upsample_rendering parameter
  else:

    # upsample_rendering of 0, 1 or 2 means we just need to pick one of the samples
    if stereo_rendering < 3:
      wav = wav[stereo_rendering, :, 0]

    # upsample_rendering of 3 means we need to convert the audio to stereo, putting sample 0 to the left, 1 to the center, and 2 to the right
    # 4 means we also want to add a delay of 20 ms for the left and 40 ms for the right channel
    else:
      wav = to_stereo(wav, 20 if stereo_rendering == 4 else 0, invert_center)

  upsampled_lengths = [ 0, 0 ]
  if combine_levels:

    available_levels = get_levels(zs)
    combined_wav = None

    for sub_level in available_levels:

      if sub_level < level:
        sub_wav = get_audio(project_name, sample_id, cut_audio, seconds_to_cut_from_start, sub_level, stereo_rendering, combine_levels, invert_center)[0]
        upsampled_lengths[sub_level] = sub_wav.shape[0] / hps.sr + seconds_to_cut_from_start
      else:
        sub_wav = wav
        # If the wav is mono, we need to convert it to stereo by using the same values for both channels
        # (Note that this is most always the case, since the original audio is always mono, and this function is likely to be called for the original level, but we're abstracting it just in case)
        if sub_wav.ndim == 1:
          sub_wav = np.stack([ sub_wav, sub_wav ], axis=1)

      if combined_wav is None:
        combined_wav = sub_wav
        print(f'Created wav of length {combined_wav.shape[0]} for level {sub_level}')
      else:
        n_to_add = sub_wav.shape[0] - combined_wav.shape[0]
        # (This might be confusing why we are subtracting the shape of combined wav from the "sub" wav, but it's because the higher level "sub" wav is the one that is being upsampled, so it's the one that needs to be longer. The entire terminology with levels going backwards while the quality goes up is confusing, but we work with what we have)
        if n_to_add > 0:
          print(f'Adding {n_to_add} samples for level {sub_level}')
          combined_wav = np.concatenate([ combined_wav, sub_wav[ -n_to_add: ] ], axis=0)

    wav = combined_wav

  print(f'Generated audio of length {len(wav)} ({ len(wav) / hps.sr } seconds); original length: {audio_length} seconds.')

  return wav, audio_length, upsampled_lengths