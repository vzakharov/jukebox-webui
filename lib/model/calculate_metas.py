from .model import Model
from .params import hps

class Metas:

  calculated_for = {}
  metas = None
  labels = None

  def calculate(artist, genre, lyrics, n_samples, discard_window):

    if discard_window > 0:
      # If there's "---\n" in the lyrics, remove everything before and including it
      cutout = '---\n'
      if lyrics and cutout in lyrics:
        lyrics = lyrics.split(cutout)[1]
        print(f'Lyrics after cutting: {lyrics}')

      print(f'Metas or n_samples have changed, recalculating the model for {artist}, {genre}, {lyrics}, {n_samples} samples...')

    Metas.metas = [dict(
      artist = artist,
      genre = genre,
      total_length = hps.sample_length,
      offset = 0,
      lyrics = lyrics,
    )] * n_samples

    Metas.labels = Model.top_prior.labeller.get_batch_labels(Metas.metas, Model.device)

    Metas.calculated_for = {
      'artist': artist,
      'genre': genre,
      'lyrics': lyrics
    }

    print('Done recalculating the metas')