from .cut import cut_z
from lib.model.params import hps
from lib.navigation.utils import get_zs
from lib.upsampling.utils import get_levels
from lib.utils import seconds_to_tokens, tokens_to_seconds
from main import vqvae
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
  # z is of shape torch.Size([1, n_tokens])
  # print(f'Loaded {filename}.z at level {level}, shape: {z.shape}')

  if cut_audio:
    z = cut_z(z, cut_audio, level)

  # Update audio_length
  audio_length = int( tokens_to_seconds(z.shape[1], level) * 100 ) / 100

  if preview_sec:
    seconds_to_cut_from_start = audio_length - abs(preview_sec) if preview_sec < 0 else preview_sec
    # For negative values, we need to replace "-" with "<" because "-" is used to indicate a range
    z = cut_z(z, f'-{seconds_to_cut_from_start}', level)
  else:
    seconds_to_cut_from_start = 0

  def decode(z):
    if z.shape[1] > 0:
      wav = vqvae.decode([ z ], start_level=level, end_level=level+1).cpu().numpy()
      # the decoded wav is of shape (n_samples, sample_length, 1). We will convert it later to (n_samples, 1 or 2 depending on stereo_rendering)
    else:
      # If the sample is empty, we need to create an empty wav of the right shape
      wav = np.zeros((z.shape[0], 0, 1))
    return wav

  # If z is longer than 30 seconds, there will likely be not enough RAM to decode it in one go
  # In this case, we'll split it into 30-second chunks (with a 5-second overlap), decode each chunk separately, and concatenate the results, crossfading the overlaps
  if z.shape[1] < seconds_to_tokens(30, level):
    wav = decode(z)
  else:
    chunk_size = seconds_to_tokens(30, level)
    overlap_size = seconds_to_tokens(5, level)
    print(f'z is too long ({z.shape[1]} tokens), splitting into chunks of {chunk_size} tokens, with a {overlap_size} token overlap')
    wav = None
    # Keep in mind that the last chunk can be shorter if the total length is not a multiple of chunk_size)
    for i in range(0, z.shape[1], chunk_size - overlap_size):

      # If this is the last chunk, make the chunk_size smaller if necessary
      overflow = i + chunk_size - z.shape[1]
      is_last_chunk = overflow > 0
      if is_last_chunk:
        chunk_size -= overflow
        # print(f'Last chunk, reduced chunk_size from {chunk_size + overflow} to {chunk_size} tokens')

      left_overlap_z = z[ :, i:i+overlap_size ]
      # print(f'Left overlap (tokens): {left_overlap_z.shape[1]}')
      left_overlap = decode(left_overlap_z)
      # print(f'Left overlap (quants): {left_overlap.shape[1]}')

      def fade(overlap, direction):
        # To fade in, we need to add 1/4 of the overlap as silence, 2/4 of the overlap as a linear ramp, and 1/4 of the overlap as full volume
        is_fade_in = direction == 'in'
        overlap_quants = overlap.shape[1]
        silence_quants = int( overlap_quants / 4 )
        ramp_quants = int( overlap_quants / 2 )
        if is_fade_in:
          overlap[:, :silence_quants, :] = 0
        else:
          overlap[:, -silence_quants:, :] = 0
        start = 0 if is_fade_in else 1
        overlap[:, silence_quants:-silence_quants, :] *= np.linspace(start, 1 - start, ramp_quants).reshape(1, -1, 1)
        return overlap

      if wav is not None:

        # Fade in the left overlap and add it to the existing wav if it's not empty (i.e. if this is not the first chunk)
        left_overlap = fade(left_overlap, 'in')
        # print(f'Faded in left overlap')
        # # Show as plot
        # plt.plot(left_overlap[0, :, 0])
        # plt.show()

        wav[ :, -left_overlap.shape[1]: ] += left_overlap
        # print(f'Added left overlap to existing wav:')
        # # Plot the resulting (faded-in + previous fade-out) overlap
        # plt.plot(wav[0, -left_overlap.shape[1]:, 0])
        # plt.show()

        print(f'Added left overlap to wav, overall shape now: {wav.shape}')

      else:
        wav = left_overlap
        print(f'Created wav with left overlap')

      # We'll also won't need right overlap for the last chunk
      main_chunk_z = z[ :, i+overlap_size: i+chunk_size-overlap_size if not is_last_chunk else i+chunk_size ]
      print(f'Main chunk (tokens): {main_chunk_z.shape[1]}')

      if main_chunk_z.shape[1] > 0:

        main_chunk = decode(main_chunk_z)
        print(f'Main chunk (quants): {main_chunk.shape[1]}')

        # Add the main chunk to the existing wav
        wav = np.concatenate([ wav, main_chunk ], axis=1)
        print(f'Added main chunk to wav, overall shape now: {wav.shape}')

      else:
        print('Main chunk is empty, skipping')
        continue

      # Fade out the right overlap, unless this is the last chunk
      if not is_last_chunk:

        right_overlap_z = z[ :, i+chunk_size-overlap_size:i+chunk_size ]
        # print(f'Right overlap (tokens): {right_overlap_z.shape[1]}')

        right_overlap = decode(right_overlap_z)
        # print(f'Right overlap (quants): {right_overlap.shape[1]}')

        right_overlap = fade(right_overlap, 'out')
        # print(f'Faded out right overlap')
        # # Show as plot
        # plt.plot(right_overlap[0, :, 0])
        # plt.show()

        # Add the right overlap to the existing wav
        wav = np.concatenate([ wav, right_overlap ], axis=1)
        # print(f'Added right overlap to wav, overall shape now: {wav.shape}')

      else:

        print(f'Last chunk, not adding right overlap')
        break

      print(f'Decoded {i+chunk_size} tokens out of {z.shape[1]}, wav shape: {wav.shape}')

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

      # def to_stereo(wav, stereo_delay_ms=0):
      def to_stereo(wav, stereo_delay_ms=0, invert_center=False):

        # A stereo wav is of form (sample_length + double the delay, 2)
        delay_quants = int( stereo_delay_ms * hps.sr / 1000 )
        stereo = np.zeros((wav.shape[1] + 2 * delay_quants, 2))
        # First let's convert the wav to [n_quants, n_samples] by getting rid of the last dimension and transposing the rest
        wav = wav[:, :, 0].T
        # print(f'Converted wav to shape {wav.shape}')
        # Take sample 0 for left channel (delayed once), 1 for both channels (non-delayed), and sample 2 for right channel (delayed twice)
        if delay_quants:
          stereo[ delay_quants: -delay_quants, 0 ] = wav[ :, 0 ]
          stereo[ 2 * delay_quants:, 1 ] = wav[ :, 2 ]
          stereo[ : -2 * delay_quants, 0 ] += wav[ :, 1 ]
          stereo[ : -2 * delay_quants, 1 ] += wav[ :, 1 ]
        else:
          stereo[ :, 0 ] = wav[ :, 0 ] + wav[ :, 1 ]
          stereo[ :, 1 ] = wav[ :, 2 ] + wav[ :, 1 ]

        if invert_center:
          stereo[ :, 1 ] *= -1

        # Now we have max amplitude of 2, so we need to divide by 2
        stereo /= 2

        # print(f'Converted to stereo with delay {stereo_delay_ms} ms, current shape: {stereo.shape}, max/min amplitudes: {np.max(stereo)}/{np.min(stereo)}')

        return stereo

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