import shutil
from datetime import datetime

import torch as t
from jukebox.sample import upsample

from lib.model.params import hps, priors, top_prior
from params import base_path

from .get_zs_from_ancestor_if_any import get_zs_from_ancestor_if_any
from .init_upsampling import init_upsampling
from .labels import get_labels
from .Upsampling import Upsampling


def start_upsampling(project_name, sample_id, artist, lyrics, genre_left, genre_center, genre_right, kill_runtime_once_done=False):

  global hps, top_prior, priors

  genres = [genre_left, genre_center, genre_right]

  print(f'Starting upsampling for {sample_id}, artist: {artist}, lyrics: {lyrics}, genres: {genres}')

  init_upsampling(project_name, sample_id, kill_runtime_once_done)

  print(f'Upsampling {sample_id} with genres {genres}')
  filename = f'{base_path}/{project_name}/{sample_id}.z'

  Upsampling.zs = t.load(filename)

  # Get the level 0/1 zs of the first upsampled ancestor (so we can continue upsampling from where we left off)
  get_zs_from_ancestor_if_any(project_name, sample_id)

  print(f'Final z shapes: {[ z.shape for z in Upsampling.zs ]}')

  # We also need to create new labels from the metas with the genres replaced accordingly
  labels = get_labels(project_name, artist, lyrics, genres)

  if type(labels)==dict:
    labels = [ prior.labeller.get_batch_labels(Upsampling.metas, 'cuda') for prior in Upsampling.upsamplers ] + [ labels ]
    print('Converted labels to list')
    # Not sure why we need to do this -- I copied this from another notebook.

  Upsampling.labels = labels

  # Create a backup of the original file, in case something goes wrong
  bak_filename = f'{filename}.{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.bak'
  shutil.copy(filename, f'{bak_filename}')
  print(f'Created backup of {filename} as {bak_filename}')

  t.save(Upsampling.zs, filename)

  Upsampling.zs = upsample(Upsampling.zs, Upsampling.labels, Upsampling.kwargs, Upsampling.priors, Upsampling.hps)