from jukebox.hparams import Hyperparams

from params import total_duration

raw_to_tokens = 128
chunk_size = 16
lower_batch_size = 16
lower_level_chunk_size = 32
hps = Hyperparams()

hps.sr = 44100
hps.levels = 3
hps.hop_fraction = [ 0.5, 0.5, 0.125 ]
hps.sample_length = int(total_duration * hps.sr // raw_to_tokens) * raw_to_tokens

device, vqvae, priors, top_prior = None, None, None, None