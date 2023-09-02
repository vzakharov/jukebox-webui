import gradio as gr

from lib.ui.elements.metas import artist, genre, lyrics
from lib.ui.elements.navigation import sample_tree, show_leafs_only
from lib.ui.elements.preview import just_the_last_n_sec
from lib.ui.elements.upsampling import genre_center_channel, genre_left_channel, genre_right_channel

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
  *generation_params, sample_tree, show_leafs_only, just_the_last_n_sec,
  genre_left_channel, genre_center_channel, genre_right_channel
]
total_audio_length = gr.Number(
  label = 'Total audio length, sec',
  elem_id = 'total-audio-length',
  interactive = False,
  visible = False
)