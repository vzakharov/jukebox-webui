from lib.upsampling.Upsampling import Upsampling
from main import base_path, get_first_upsampled_ancestor_zs, hps, load_top_prior, priors, raw_to_tokens, top_prior, vqvae


import torch as t
from jukebox.hparams import setup_hparams
from jukebox.make_models import make_prior
from jukebox.sample import upsample
from jukebox.utils.torch_utils import empty_cache


import os
import shutil
from datetime import datetime


def start_upsampling(project_name, sample_id, artist, lyrics, genre_left, genre_center, genre_right, kill_runtime_once_done=False):

  global hps, top_prior, priors

  genres = [genre_left, genre_center, genre_right]

  print(f'Starting upsampling for {sample_id}, artist: {artist}, lyrics: {lyrics}, genres: {genres}')

  Upsampling.project = project_name
  Upsampling.sample_id = sample_id

  Upsampling.running = True
  Upsampling.status_markdown = "Loading the upsampling models..."

  Upsampling.level = 1

  Upsampling.kill_runtime_once_done = kill_runtime_once_done

  print(f'Upsampling {sample_id} with genres {genres}')
  filename = f'{base_path}/{project_name}/{sample_id}.z'

  Upsampling.zs = t.load(filename)

  # Get the level 0/1 zs of the first upsampled ancestor (so we can continue upsampling from where we left off)
  for i in range( len(Upsampling.zs) ):
    if i == 2:
      Upsampling.zs[i] = Upsampling.zs[i][0].repeat(3, 1)
    elif Upsampling.zs[i].shape[0] != 3:
      # If there are no upsampled ancestors, replace with an empty tensor
      first_upsampled_ancestor = get_first_upsampled_ancestor_zs(project_name, sample_id)
      if not first_upsampled_ancestor:
        print(f'No upsampled ancestors found for {sample_id}, using empty tensors')
        Upsampling.zs[i] = t.empty( (3, 0), dtype=t.int64 ).cuda()
      else:
        print(f'Using first upsampled ancestor zs for {sample_id}')
        Upsampling.zs[i] = first_upsampled_ancestor[i]

  print(f'Final z shapes: {[ z.shape for z in Upsampling.zs ]}')

  # We also need to create new labels from the metas with the genres replaced accordingly
  Upsampling.metas = [ dict(
    artist = artist,
    genre = genre,
    total_length = hps.sample_length,
    offset = 0,
    lyrics = lyrics,
  ) for genre in genres ]

  if not Upsampling.labels:

    # Search for labels under [project_name]/[project_name].labels
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

  labels = Upsampling.labels

  upsamplers = [ make_prior(setup_hparams(prior, dict()), vqvae, 'cpu') for prior in priors[:-1] ]

  if type(labels)==dict:
    labels = [ prior.labeller.get_batch_labels(Upsampling.metas, 'cuda') for prior in upsamplers ] + [ labels ]
    print('Converted labels to list')
    # Not sure why we need to do this -- I copied this from another notebook.

  Upsampling.labels = labels

  # Create a backup of the original file, in case something goes wrong
  bak_filename = f'{filename}.{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.bak'
  shutil.copy(filename, f'{bak_filename}')
  print(f'Created backup of {filename} as {bak_filename}')

  t.save(Upsampling.zs, filename)

  Upsampling.params = [
    dict(temp=0.99, fp16=True, max_batch_size=16, chunk_size=32),
    dict(temp=0.99, fp16=True, max_batch_size=16, chunk_size=32),
    None
  ]

  Upsampling.hps = hps

  # Set hps.n_samples to 3, because we need 3 samples for each level
  Upsampling.hps.n_samples = 3

  # Set hps.sample_length to the actual length of the sample
  Upsampling.hps.sample_length = Upsampling.zs[2].shape[1] * raw_to_tokens

  # Set hps.name to our project directory
  Upsampling.hps.name = f'{base_path}/{project_name}'

  Upsampling.priors = [*upsamplers, None]
  Upsampling.zs = upsample(Upsampling.zs, Upsampling.labels, Upsampling.params, Upsampling.priors, Upsampling.hps)