import numpy as np

from lib.model.params import hps
from lib.upsampling.utils import get_levels

from .get_audio import get_audio


def combine_levels(project_name, sample_id, cut_audio, level, stereo_rendering, invert_center, zs, seconds_to_cut_from_start, wav, upsampled_lengths):
  available_levels = get_levels(zs)
  combined_wav = None

  for sub_level in available_levels:
    if sub_level < level:
      sub_wav = get_audio(project_name, sample_id, cut_audio, seconds_to_cut_from_start, sub_level, stereo_rendering, True, invert_center)[0]
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
  return wav