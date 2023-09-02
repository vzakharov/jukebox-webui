import gradio as gr

import UI.general
import UI.main
import UI.metas
import UI.misc
import UI.navigation
import UI.project
import UI.upsampling

from .app import app
from .ui.on_load import on_load
from .utils import read, set_browser_timezone


def on_app_load():
  app.load(
    on_load,
    inputs = [ gr.Textbox(visible=False), gr.Textbox(visible=False), gr.Textbox(visible=False) ],
    outputs = [
      UI.general.project_name, UI.navigation.routed_sample_id,
      UI.metas.artist, UI.metas.genre_dropdown,
      UI.misc.getting_started_column, UI.misc.separate_tab_warning, UI.misc.separate_tab_link, UI.main.main_window,
      UI.upsampling.genre_left_channel, UI.upsampling.genre_center_channel, UI.upsampling.genre_right_channel
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