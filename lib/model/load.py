import sys

import jukebox
from jukebox.hparams import setup_hparams
from jukebox.make_models import MODELS, make_prior, make_vqvae
from jukebox.utils.dist_utils import setup_dist_from_mpi
from jukebox.utils.torch_utils import empty_cache

from .params import device, hps, priors, top_prior, vqvae
from lib.monkey_patches.load_audio import monkey_patched_load_audio
from lib.monkey_patches.load_checkpoint import monkey_patched_load_checkpoint
from lib.monkey_patches.sample_level import monkey_patched_sample_level
from lib.upsampling.Upsampling import Upsampling
from lib.upsampling.utils import keep_upsampling_after_restart
from params import reload_all, total_duration


def load_top_prior(priors):
  global top_prior, vqvae, device

  print('Loading top prior')
  top_prior = make_prior(setup_hparams(priors[-1], dict()), vqvae, device)
  
def load_model():

  global device, vqvae, priors, top_prior
  
  if '--no-load' in sys.argv:
    print("🚫 Skipping model loading")
    return

  rank, local_rank, device = setup_dist_from_mpi()
  print(f'Dist setup: rank={rank}, local_rank={local_rank}, device={device}')


  print('Monkey patching Jukebox methods...')

  jukebox.make_models.load_checkpoint = monkey_patched_load_checkpoint
  jukebox.utils.audio_utils.load_audio = monkey_patched_load_audio
  jukebox.sample.sample_level = monkey_patched_sample_level

  reload_prior = False #param{type:'boolean'}

  if Upsampling.running:
    print('''
      !!! APP SET FOR UPSAMPLING !!!
      To use the app for composing, stop execution, create a new cell and run the following code:
      Upsampling.started = False
      Then run the main cell again.
    ''')
  else:

    if not keep_upsampling_after_restart:

      try:
        vqvae, priors, top_prior

        assert total_duration == calculated_duration and not reload_prior and not reload_all
        print('Model already loaded.')

      except:

        print(f'Loading vqvae and top_prior for duration {total_duration}...')
        vqvae, *priors = MODELS['5b_lyrics']
        vqvae = make_vqvae(setup_hparams(vqvae, dict(sample_length = hps.sample_length)), device)
        load_top_prior(priors)
        calculated_duration = total_duration