import lib.ui.components as UI

import gradio as gr

def render_genres_for_upsampling():
  with gr.Accordion('Genres for upsampling (optional)', open = False):
    with gr.Accordion('What is this?', open = False):
      gr.Markdown('''
        The tool will generate three upsamplings of the selected sample, which will then be panned to the left, center, and right, respectively. Choosing different genres for each of the three upsamplings will result in a more diverse sound between them, thus enhancing the (pseudo-)stereo effect. 
        A good starting point is to have a genre that emphasizes vocals (e.g. `Pop`) for the center channel, and two similar but different genres for the left and right channels (e.g. `Rock` and `Metal`).
        If you don’t want to use this feature, simply select the same genre for all three upsamplings.
      ''')

    with gr.Row():
      for input in [ UI.genre_for_upsampling_left_channel, UI.genre_for_upsampling_center_channel, UI.genre_for_upsampling_right_channel ]:
        input.render()

    UI.kill_runtime_once_done.render()