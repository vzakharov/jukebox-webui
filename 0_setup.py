if True:

  from google.colab import drive
  drive.mount('/content/gdrive')
  !pip install git+https://github.com/openai/jukebox.git
  import time
  import jukebox
  import torch as t
  import librosa
  import glob
  import os
  import numpy as np
  import re
  from datetime import datetime, timedelta, timezone
  from IPython.display import Audio, display
  from jukebox.make_models import make_vqvae, make_prior, MODELS, make_model
  from jukebox.hparams import Hyperparams, setup_hparams
  from jukebox import sample
  from jukebox.sample import sample_single_window, _sample, \
                            sample_partial_window, upsample, \
                            load_prompts
  from jukebox.utils.dist_utils import setup_dist_from_mpi
  from jukebox.utils.torch_utils import empty_cache
  rank, local_rank, device = setup_dist_from_mpi()

  # Monkey patch sample_single_window to write the zs to a temporary file
  original_sample_single_window = sample.sample_single_window

  last_sampled = None

  def patched_sample_single_window(*args, **kwargs):
    global zs, t, hps, project_name, priors, chunk_size
    zs = original_sample_single_window(*args, **kwargs)

    try:

      # Find the level being sampled
      level = 0
      for i in range(len(zs)):
        if zs[i].size != 0:
          level = i
      
      # # Print the number of seconds already generated at this level
      # print(f'Generated {seconds_to_tokens(zs[level].shape[1], hps.sr, None, chunk_size) / hps.sr} seconds at level {level}')
      # # Print the remaining number of seconds to be generated at this level
      # tokens_left = hps.n_ctx - zs[level].shape[1]
      # print(f'Remaining: {seconds_to_tokens(tokens_left, hps.sr, None, chunk_size) / hps.sr} seconds')

      # if last_sampled:
      #   # Calculate the time elapsed since the last sample
      #   time_elapsed = datetime.now() - last_sampled
      #   # Calculate the time per token
      #   time_per_token = time_elapsed / hps.n_ctx
      #   # Calculate the remaining time
      #   remaining_time = time_per_token * tokens_left
      #   print(f'Estimated time remaining: {remaining_time}')

      # last_sampled = datetime.now()

      # Save with a timestamp
      filename = f'{project_name}-tmp-{datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M-%S")}.zs'
      # If 'tmp' folder doesn't exist, create it
      if not os.path.exists(f'{hps.name}/tmp'):
        os.makedirs(f'{hps.name}/tmp')
      t.save(zs, f'{hps.name}/tmp/{filename}')
      # If there's a file called 'render' in the project folder, render the current zs (zs_to_wavs)
      if os.path.exists(f'{hps.name}/render'):
        print(f'Rendering {filename}...')
        os.remove(f'{hps.name}/render')
        zs_to_wavs(level=level)
        print(f'Done; don’t forget to create a file called “render” to render again.')
    
    except Exception as e:
      print(e)
    
    finally:
      return zs

  sample.sample_single_window = patched_sample_single_window

  model = "5b_lyrics"
  hps = Hyperparams()
  hps.hop_fraction = [.5, .5, .125] 
  hps.levels = 3
  hps.sr = 44100

  chunk_size = 16
  max_batch_size = 3
  lower_batch_size = 16
  lower_level_chunk_size = 32

  raw_to_tokens = 128
  sample_length_in_seconds = 200         #@param{type:'number'}

  hps.sample_length = (int(sample_length_in_seconds*hps.sr)//raw_to_tokens)*raw_to_tokens

  vqvae, *priors = MODELS[model]
  vqvae = make_vqvae(setup_hparams(vqvae, dict(sample_length = hps.sample_length)), device)
  top_prior = make_prior(setup_hparams(priors[-1], dict()), vqvae, device)

  assert hps.sample_length >= top_prior.n_ctx*top_prior.raw_to_tokens, f'Please choose a larger sampling rate'

  zs_old_filename = None
  old_choice = None