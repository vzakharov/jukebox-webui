from .ui.UI import UI
from .ui.on_load import on_load
from .utils import read, set_browser_timezone
from .app import app


import gradio as gr


def on_app_load():
  app.load(
    on_load,
    inputs = [ gr.Textbox(visible=False), gr.Textbox(visible=False), gr.Textbox(visible=False) ],
    outputs = [
      UI.project_name, UI.routed_sample_id, UI.artist, UI.genre_dropdown, UI.getting_started_column, UI.separate_tab_warning, UI.separate_tab_link, UI.main_window,
      UI.genre_for_upsampling_left_channel, UI.genre_for_upsampling_center_channel, UI.genre_for_upsampling_right_channel
    ],
    api_name = 'initialize',
    _js = read('frontend-on-load.js')
  )

  app.load(
    inputs = gr.Number( visible = False ),
    outputs = None,
    _js = '() => [ new Date().getTimezoneOffset() ]',
    fn = set_browser_timezone
  )