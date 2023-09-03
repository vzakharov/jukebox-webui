from jukebox.hparams import setup_hparams
from jukebox.make_models import make_prior

from lib.model.params import hps, raw_to_tokens
from lib.model.params import priors, vqvae
from params import base_path

from .Upsampling import Upsampling


def init_upsampling(project_name, sample_id, kill_runtime_once_done):
  Upsampling.project = project_name
  Upsampling.sample_id = sample_id

  Upsampling.running = True
  Upsampling.status_markdown = "Loading the upsampling models..."

  Upsampling.level = 1

  Upsampling.kill_runtime_once_done = kill_runtime_once_done

  Upsampling.upsamplers = [ make_prior(setup_hparams(prior, dict()), vqvae, 'cpu') for prior in priors[:-1] ]

  Upsampling.kwargs = [
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

  Upsampling.priors = [*Upsampling.upsamplers, None]