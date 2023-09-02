from .params import hps
from main import device, top_prior

calculated_metas = {}
metas = None
labels = None

def calculate_metas(artist, genre, lyrics, n_samples, discard_window):

  global calculated_metas, labels, metas

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