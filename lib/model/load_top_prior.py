from .params import device, top_prior, vqvae


from jukebox.hparams import setup_hparams
from jukebox.make_models import make_prior


def load_top_prior(priors):
  global top_prior, vqvae, device

  print('Loading top prior')
  top_prior = make_prior(setup_hparams(priors[-1], dict()), vqvae, device)