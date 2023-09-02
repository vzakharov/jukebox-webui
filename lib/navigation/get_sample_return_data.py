from .get_children import get_children
from .get_parent import get_parent
from lib.ui.elements.audio import current_chunks
from lib.ui.elements.navigation import picked_sample_updated
from lib.ui.elements.sample import go_to_children, go_to_parent, sample_box


import gradio as gr

import random

def get_sample_return_data(project_name, sample_id, total_audio_length, upsampled_lengths, chunk_filenames):
    return {
    current_chunks: chunk_filenames,
    total_audio_length: total_audio_length,
    go_to_children: gr.update(
      visible = len(get_children(project_name, sample_id)) > 0
    ),
    go_to_parent: gr.update(
      visible = get_parent(project_name, sample_id) is not None
    ),
    sample_box: gr.update(
      visible = True
    ),
    upsampled_lengths: ','.join([str(length) for length in upsampled_lengths]),
    # Random number for picked sample updated flag
    picked_sample_updated: random.random(),
  }