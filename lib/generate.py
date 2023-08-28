from lib.navigation.get_samples import get_samples
from lib.navigation.get_first_free_index import get_first_free_index
from lib.navigation.utils import get_prefix
from lib.ui.UI import UI
from lib.utils import seconds_to_tokens
from main import device, priors, top_prior

import gradio as gr
import torch as t
from jukebox.sample import sample_partial_window

import random
from datetime import datetime

from params import base_path, total_duration
from lib.model.params import chunk_size, hps, lower_batch_size, lower_level_chunk_size, raw_to_tokens

calculated_metas = {}

def generate(project_name, parent_sample_id, show_leafs_only, artist, genre, lyrics, n_samples, temperature, generation_length, discard_window):

  print(f'Generating {n_samples} sample(s) of {generation_length} sec each for project {project_name}...')

  global total_duration
  global calculated_metas
  global hps, raw_to_tokens, chunk_size, lower_batch_size, lower_level_chunk_size
  global top_prior, device, priors
  global metas, labels

  hps.n_samples = n_samples

  # If metas or n_samples have changed, recalculate the metas
  if calculated_metas != dict( artist = artist, genre = genre, lyrics = lyrics ) or len(metas) != n_samples:

    if discard_window > 0:
      # If there's "---\n" in the lyrics, remove everything before and including it
      cutout = '---\n'
      if lyrics and cutout in lyrics:
        lyrics = lyrics.split(cutout)[1]
        print(f'Lyrics after cutting: {lyrics}')

      print(f'Metas or n_samples have changed, recalculating the model for {artist}, {genre}, {lyrics}, {n_samples} samples...')

    metas = [dict(
      artist = artist,
      genre = genre,
      total_length = hps.sample_length,
      offset = 0,
      lyrics = lyrics,
    )] * n_samples

    labels = top_prior.labeller.get_batch_labels(metas, device)

    calculated_metas = {
      'artist': artist,
      'genre': genre,
      'lyrics': lyrics
    }

    print('Done recalculating the model')

  print(f'Generating {generation_length} seconds for {project_name}...')

  if parent_sample_id:

    zs = t.load(f'{base_path}/{project_name}/{parent_sample_id}.z')
    print(f'Loaded parent sample {parent_sample_id} of shape {[ z.shape for z in zs ]}')
    zs = [ z[0].repeat(n_samples, 1) for z in zs ]
    print(f'Converted to shape {[ z.shape for z in zs ]}')

    if discard_window > 0:

      discarded_zs = [ z[:, :seconds_to_tokens(discard_window)] for z in zs ]
      zs = [ z[:, seconds_to_tokens(discard_window):] for z in zs ]
      print(f'Discarded the first {discard_window} seconds, now zs are of shape {[ z.shape for z in zs ]}')

  else:
    zs = [ t.zeros(n_samples, 0, dtype=t.long, device='cuda') for _ in range(3) ]
    print('No parent sample or primer provided, starting from scratch')

  tokens_to_sample = seconds_to_tokens(generation_length)
  sampling_kwargs = dict(
    temp=temperature, fp16=True, max_batch_size=lower_batch_size,
    chunk_size=lower_level_chunk_size
  )

  print(f'zs: {[ z.shape for z in zs ]}')
  zs = sample_partial_window(zs, labels, sampling_kwargs, 2, top_prior, tokens_to_sample, hps)
  print(f'Generated zs of shape {[ z.shape for z in zs ]}')

  if discard_window > 0:
    zs = [ t.cat([ discarded_zs[i], zs[i] ], dim=1) for i in range(3) ]
    print(f'Concatenated cutout zs of shape {[ z.shape for z in discarded_zs ]} with generated zs of shape {[ z.shape for z in zs ]}')

  prefix = get_prefix(project_name, parent_sample_id)
  # For each sample, write the z (a subarray of zs)

  try:
    first_new_child_index = get_first_free_index(project_name, parent_sample_id)
  except Exception as e:
    print(f'Something went wrong: {e}')
    first_new_child_index = random.randrange(1e6, 1e7)
    print(f'Using random index {first_new_child_index} as a fallback')

  for i in range(n_samples):
    id = f'{prefix}{first_new_child_index + i}'
    filename = f'{base_path}/{project_name}/{id}'

    # zs is a list of 3 tensors, each of shape (n_samples, n_tokens)
    # To write the z for a single sample, we need to take a subarray of each tensor
    this_sample_zs = [ z[i:i+1] for z in zs ]

    t.save(this_sample_zs, f'{filename}.z')
    print(f'Wrote {filename}.z')

  return {
    UI.sample_tree: gr.update(
      choices = get_samples(project_name, show_leafs_only),
      value = id
    ),
    UI.generation_progress: f'Generation completed at {datetime.now().strftime("%H:%M:%S")}'
  }