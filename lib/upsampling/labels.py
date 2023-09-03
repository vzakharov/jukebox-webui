import os

import torch as t
from jukebox.utils.torch_utils import empty_cache

from lib.model.load_top_prior import load_top_prior
from lib.model.params import hps
from .Upsampling import Upsampling
from params import base_path

from .Upsampling import Upsampling

def prepare_labels(project_name):
  labels_path = f'{base_path}/{project_name}/{project_name}.labels'

  should_reload_labels = True

  if os.path.exists(labels_path):
    Upsampling.labels, stored_metas = t.load(labels_path)
    print(f'Loaded labels from {labels_path}')
    # Make sure the metas match
    if stored_metas == Upsampling.metas:
      print('Metas match, not reloading labels')
      should_reload_labels = False
    else:
      print(f'Metas do not match, reloading labels. Stored metas: {stored_metas}, current metas: {Upsampling.metas}')

  if should_reload_labels:
    try:
      assert top_prior
    except:
      load_top_prior()

    Upsampling.labels = top_prior.labeller.get_batch_labels(Upsampling.metas, 'cuda')
    print('Calculated new labels from top prior')

    t.save([ Upsampling.labels, Upsampling.metas ], labels_path)
    print(f'Saved labels and metas to {labels_path}')

    # We need to delete the top_prior object and empty the cache, otherwise we'll get an OOM error
    del top_prior
    empty_cache()

def get_labels(project_name, artist, lyrics, genres):
  Upsampling.metas = [ dict(
    artist = artist,
    genre = genre,
    total_length = hps.sample_length,
    offset = 0,
    lyrics = lyrics,
  ) for genre in genres ]

  if not Upsampling.labels:
    prepare_labels(project_name)

  labels = Upsampling.labels
  return labels