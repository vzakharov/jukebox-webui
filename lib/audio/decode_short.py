from lib.model.model import Model

import numpy as np

def decode_short(z, level):
  if z.shape[1] > 0:
    wav = Model.vqvae.decode([ z ], start_level=level, end_level=level+1).cpu().numpy()
    # the decoded wav is of shape (n_samples, sample_length, 1). We will convert it later to (n_samples, 1 or 2 depending on stereo_rendering)
  else:
    # If the sample is empty, we need to create an empty wav of the right shape
    wav = np.zeros((z.shape[0], 0, 1))
  return wav