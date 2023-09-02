from lib.utils import seconds_to_tokens
from .decode_short import decode_short


import numpy as np


def decode(z, level):

  if z.shape[1] < seconds_to_tokens(30, level):
    return decode_short(z, level)

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
    left_overlap = decode_short(left_overlap_z)
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
      main_chunk = decode_short(main_chunk_z)
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

      right_overlap = decode_short(right_overlap_z)
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
  return wav