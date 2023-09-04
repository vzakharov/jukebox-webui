import os

import gradio as gr

from params import base_path, debug_gradio, share_gradio

from .api import define_get_projects_api
from .app import app
from .model.load import load_model
from .on_app_load import on_app_load
from .ui.blocks.getting_started import render_getting_started
from .ui.blocks.sidebar.sidebar import render_sidebar
from .ui.blocks.workspace.workspace import render_workspace
from .ui.elements.main import main_window
from .ui.elements.misc import separate_tab_link, separate_tab_warning


def run():

  load_model()

  # If the base folder doesn't exist, create it
  if not os.path.isdir(base_path):
    os.makedirs(base_path)

  with app:

    # Render an invisible checkbox group to enable loading list of projects via API
    define_get_projects_api()

    with separate_tab_warning.render():

      separate_tab_link.render()

      gr.Button('Click here to open the UI', variant = 'primary' ).click( inputs = separate_tab_link, outputs = None, fn = None,
        _js = "link => window.open(link, '_blank')"
      )

    with main_window.render():

      render_sidebar()
      render_getting_started()
      render_workspace()

    on_app_load()

    app.launch( share = share_gradio, debug = debug_gradio )