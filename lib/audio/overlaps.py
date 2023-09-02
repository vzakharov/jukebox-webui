import numpy as np

from .decode_short import decode_short

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

def add_right_overlap(wav, z, chunk_size, overlap_size, i):
  right_overlap_z = z[ :, i+chunk_size-overlap_size:i+chunk_size ]
  right_overlap = decode_short(right_overlap_z)
  right_overlap = fade(right_overlap, 'out')

  wav = np.concatenate([ wav, right_overlap ], axis=1)
  return wav

def add_left_overlap(wav, z, overlap_size, i):
  left_overlap_z = z[ :, i:i+overlap_size ]
  left_overlap = decode_short(left_overlap_z)

  if wav is not None:
    left_overlap = fade(left_overlap, 'in')
    wav[ :, -left_overlap.shape[1]: ] += left_overlap

    print(f'Added left overlap to wav, overall shape now: {wav.shape}')

  else:
    wav = left_overlap
    print(f'Created wav with left overlap')
  return wav