import torch as t
from jukebox.sample import sample_partial_window

from lib.utils import seconds_to_tokens

from .metas import Metas
from .model import Model
from .params import hps, lower_batch_size, lower_level_chunk_size


def generate_zs(zs, generation_length, temperature=0.99, discard_window=0, discarded_zs=None):

  tokens_to_sample = seconds_to_tokens(generation_length)
  sampling_kwargs = dict(
    temp=temperature, fp16=True, max_batch_size=lower_batch_size,
    chunk_size=lower_level_chunk_size
  )

  print(f'zs: {[ z.shape for z in zs ]}')
  zs = sample_partial_window(zs, Metas.labels, sampling_kwargs, 2, Model.top_prior, tokens_to_sample, hps)
  print(f'Generated zs of shape {[ z.shape for z in zs ]}')

  if discard_window > 0:
    zs = [ t.cat([ discarded_zs[i], zs[i] ], dim=1) for i in range(3) ]
    print(f'Concatenated cutout zs of shape {[ z.shape for z in discarded_zs ]} with generated zs of shape {[ z.shape for z in zs ]}')
    
  return zs