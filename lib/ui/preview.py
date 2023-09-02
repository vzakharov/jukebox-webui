from lib.navigation.get_sample import get_sample
import gradio as gr
from .UI import UI

preview_inputs = [
    UI.project_name, UI.picked_sample, UI.cut_audio_specs, UI.preview_just_the_last_n_sec,
    UI.upsample_rendering, UI.combine_upsampling_levels, UI.invert_center_channel
]
get_preview_args = lambda force_reload: dict(
  inputs = [
    *preview_inputs, gr.State(force_reload)
  ],
  outputs = [
    UI.sample_box, UI.current_chunks, #UI.generated_audio,
    UI.total_audio_length, UI.upsampled_lengths,
    UI.go_to_children_button, UI.go_to_parent_button,
    UI.picked_sample_updated
  ],
  fn = get_sample,
)

default_preview_args = get_preview_args(False)