import numpy as np
from .overlaps import add_left_overlap, add_right_overlap

from lib.utils import seconds_to_tokens

from .decode_short import decode_short

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

    wav = add_left_overlap(wav, z, level, overlap_size, i)

    # We'll also won't need right overlap for the last chunk
    main_chunk_z = z[ :, i+overlap_size: i+chunk_size-overlap_size if not is_last_chunk else i+chunk_size ]
    print(f'Main chunk (tokens): {main_chunk_z.shape[1]}')

    if main_chunk_z.shape[1] > 0:
      main_chunk = decode_short(main_chunk_z, level)
      print(f'Main chunk (quants): {main_chunk.shape[1]}')

      # Add the main chunk to the existing wav
      wav = np.concatenate([ wav, main_chunk ], axis=1)
      print(f'Added main chunk to wav, overall shape now: {wav.shape}')

    else:
      print('Main chunk is empty, skipping')
      continue

    # Fade out the right overlap, unless this is the last chunk
    if not is_last_chunk:
      wav = add_right_overlap(wav, z, level, chunk_size, overlap_size, i)

    else:
      print(f'Last chunk, not adding right overlap')
      break

    print(f'Decoded {i+chunk_size} tokens out of {z.shape[1]}, wav shape: {wav.shape}')
  return wav