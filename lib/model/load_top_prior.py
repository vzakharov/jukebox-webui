from jukebox.hparams import setup_hparams
from jukebox.make_models import make_prior

from .model import Model


def load_top_prior(priors):
  print('Loading top prior')
  Model.top_prior = make_prior(setup_hparams(priors[-1], dict()), Model.vqvae, Model.device)