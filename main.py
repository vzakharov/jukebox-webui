import os
import sys

import gradio as gr

from lib.api import define_get_projects_api
from lib.app import app
from lib.model.load import load_model
from lib.model.params import hps
from lib.on_app_load import on_app_load
from lib.ui.blocks.getting_started import render_getting_started
from lib.ui.blocks.sidebar.sidebar import render_sidebar
from lib.ui.blocks.workspace.workspace import render_workspace
from params import base_path, debug_gradio, share_gradio
from lib.ui.elements.misc import (browser_timezone, main_window, separate_tab_link,
                     separate_tab_warning)

print("Launch arguments:", sys.argv)

if '--no-load' in sys.argv:
  print("🚫 Skipping model loading")
  pass

else:

  device, browser_timezone, keep_upsampling_after_restart, vqvae, priors, top_prior, sample_id_to_restart_upsampling_with = load_model(hps)

# If the base folder doesn't exist, create it
if not os.path.isdir(base_path):
  os.makedirs(base_path)

with app:

  browser_timezone.render()

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

