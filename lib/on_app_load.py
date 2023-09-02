import gradio as gr

from UI.general import project_name
from UI.main import main_window
from UI.metas import artist, genre_dropdown
from UI.misc import getting_started_column, separate_tab_warning, separate_tab_link
from UI.navigation import routed_sample_id
from UI.upsampling import genre_left_channel, genre_center_channel, genre_right_channel

from .app import app
from .ui.on_load import on_load
from .utils import read, set_browser_timezone


def on_app_load():
  app.load(
    on_load,
    inputs = [ gr.Textbox(visible=False), gr.Textbox(visible=False), gr.Textbox(visible=False) ],
    outputs = [
      project_name, routed_sample_id,
      artist, genre_dropdown,
      getting_started_column, separate_tab_warning, separate_tab_link, main_window,
      genre_left_channel, genre_center_channel, genre_right_channel
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