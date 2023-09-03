from lib.model.params import hps

import numpy as np

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