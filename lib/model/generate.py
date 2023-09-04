from datetime import datetime

import gradio as gr

from lib.navigation.get_samples import get_samples
from lib.navigation.name_generations import name_generations
from lib.ui.elements.misc import generation_progress
from lib.ui.elements.navigation import sample_tree
from params import total_duration

from .calculate_metas import calculate_metas, calculated_metas, labels, metas
from .generate_zs import generate_zs
from .params import (chunk_size, hps, lower_batch_size, lower_level_chunk_size,
                     raw_to_tokens)
from .prepare_zs import prepare_zs


def generate(project_name, parent_sample_id, show_leafs_only, artist, genre, lyrics, n_samples, temperature, generation_length, discard_window):

  print(f'Generating {n_samples} sample(s) of {generation_length} sec each for project {project_name}...')

  global total_duration
  global calculated_metas
  global hps, raw_to_tokens, chunk_size, lower_batch_size, lower_level_chunk_size
  global metas, labels

  hps.n_samples = n_samples

  # If metas or n_samples have changed, recalculate the metas
  if calculated_metas != dict( artist = artist, genre = genre, lyrics = lyrics ) or len(metas) != n_samples:
    calculate_metas(artist, genre, lyrics, n_samples, discard_window)

  print(f'Generating {generation_length} seconds for {project_name}...')

  zs, discarded_zs = prepare_zs(project_name, parent_sample_id, n_samples, discard_window)

  zs = generate_zs(zs, generation_length, temperature, discard_window, discarded_zs)

  last_generated_id = name_generations(project_name, parent_sample_id, n_samples, zs)

  return {
    sample_tree: gr.update(
      choices = get_samples(project_name, show_leafs_only),
      value = last_generated_id
    ),
    generation_progress: f'Generation completed at {datetime.now().strftime("%H:%M:%S")}'
  }

