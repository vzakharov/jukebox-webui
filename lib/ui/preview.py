import gradio as gr

# import UI.audio
# import UI.general
# import UI.navigation
# import UI.preview
# import UI.project
# import UI.sample
# import UI.upsampling
from UI.audio import current_chunks
from UI.general import project_name
from UI.navigation import picked_sample, picked_sample_updated
from UI.preview import cut_audio_specs, just_the_last_n_sec
from UI.project import total_audio_length
from UI.sample import go_to_children, go_to_parent, sample_box
from UI.upsampling import combine_levels, invert_center_channel, render_mode, upsampled_lengths
from lib.navigation.get_sample import get_sample

preview_inputs = [
  # UI.general.project_name, UI.navigation.picked_sample, UI.preview.cut_audio_specs, UI.preview.just_the_last_n_sec,
  # UI.upsampling.render_mode, UI.upsampling.combine_levels, UI.upsampling.invert_center_channel
  project_name, picked_sample, cut_audio_specs, just_the_last_n_sec,
  render_mode, combine_levels, invert_center_channel
]
get_preview_args = lambda force_reload: dict(
  inputs = [
    *preview_inputs, gr.State(force_reload)
  ],
  outputs = [
    # UI.sample.sample_box, UI.audio.current_chunks,
    # UI.project.total_audio_length, UI.upsampling.upsampled_lengths,
    # UI.sample.go_to_children, UI.sample.go_to_parent,
    # UI.navigation.picked_sample_updated
    sample_box, current_chunks,
    total_audio_length, upsampled_lengths,
    go_to_children, go_to_parent,
    picked_sample_updated
  ],
  fn = get_sample,
)

default_preview_args = get_preview_args(False)