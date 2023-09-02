import torch as t
from jukebox.utils.torch_utils import empty_cache

from lib.model.params import hps
from lib.navigation.zs import get_zs
from lib.utils import seconds_to_tokens, tokens_to_seconds
from params import base_path

from .combine_levels import combine_levels
from .cut import cut_z
from .decode import decode
from .decode_short import decode_short
from .to_stereo import to_stereo


def get_audio(project_name, sample_id, cut_audio, preview_sec, level=None, stereo_rendering=3, should_combine_levels=True, invert_center=False):

  print(f'Generating audio for {project_name}/{sample_id} (level {level}, stereo rendering {stereo_rendering}, combine levels {should_combine_levels})')
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
  if should_combine_levels:

    wav = combine_levels(project_name, sample_id, cut_audio, level, stereo_rendering, invert_center, zs, seconds_to_cut_from_start, wav, upsampled_lengths)

  print(f'Generated audio of length {len(wav)} ({ len(wav) / hps.sr } seconds); original length: {audio_length} seconds.')

  return wav, audio_length, upsampled_lengths