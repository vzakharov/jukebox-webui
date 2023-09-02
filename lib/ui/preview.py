import gradio as gr

from UI.audio import current_chunks
from UI.general import project_name
from UI.navigation import picked_sample, picked_sample_updated
from UI.preview import cut_audio_specs, just_the_last_n_sec
from UI.project import total_audio_length
from UI.sample import go_to_children, go_to_parent, sample_box
from UI.upsampling import combine_levels, invert_center_channel, render_mode, upsampled_lengths

from lib.navigation.get_sample import get_sample

preview_inputs = [
  project_name, picked_sample, cut_audio_specs, just_the_last_n_sec,
  render_mode, combine_levels, invert_center_channel
]
get_preview_args = lambda force_reload: dict(
  inputs = [
    *preview_inputs, gr.State(force_reload)
  ],
  outputs = [
    sample_box, current_chunks,
    total_audio_length, upsampled_lengths,
    go_to_children, go_to_parent,
    picked_sample_updated
  ],
  fn = get_sample,
)

default_preview_args = get_preview_args(False)