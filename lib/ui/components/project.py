import gradio as gr

from lib.ui.components.metas import artist, genre, lyrics
from lib.ui.components.navigation import sample_tree, show_leafs_only
from lib.ui.components.preview import preview_just_the_last_n_sec
from lib.ui.components.upsampling import genre_for_upsampling_center_channel, genre_for_upsampling_left_channel, genre_for_upsampling_right_channel


n_samples = gr.Slider(
  label = 'Number of samples',
  minimum = 1,
  maximum = 5,
  step = 1
)
max_n_samples = gr.Number(
  visible = False
)
temperature = gr.Slider(
  label = 'Temperature',
  minimum = 0.9,
  maximum = 1.1,
  step = 0.005
)
generation_length = gr.Slider(
  label = 'Generation length, sec',
  minimum = 0.5,
  maximum = 10,
  step = 0.1
)
generation_discard_window = gr.Slider(
  label = 'Generation discard window, sec',
  minimum = 0,
  maximum = 200,
  step = 1
)
generation_params = [ artist, genre, lyrics, n_samples, temperature, generation_length, generation_discard_window ]
project_settings = [
  *generation_params, sample_tree, show_leafs_only, preview_just_the_last_n_sec,
  genre_for_upsampling_left_channel, genre_for_upsampling_center_channel, genre_for_upsampling_right_channel
]