import hashlib
import os
import sys
import urllib.request

import gradio as gr
from jukebox.sample import load_prompts

from lib.app import app
from lib.api import set_get_projects_api
from lib.cut import cut_zs
from lib.model.load import load_model
from lib.model.params import hps, set_hyperparams
from lib.app import app_layout
from lib.ui.UI import UI
from lib.on_app_load import on_app_load
from params import GITHUB_SHA, base_path, debug_gradio, share_gradio
from lib.ui.components.sidebar.sidebar import render_sidebar
from lib.ui.components.getting_started import render_getting_started
from lib.ui.components.workspace.workspace import render_workspace

print("Launch arguments:", sys.argv)

if '--no-load' in sys.argv:
  print("🚫 Skipping model loading")
  pass

else:

  device, browser_timezone, keep_upsampling_after_restart, vqvae, priors, top_prior = load_model(hps)

# If the base folder doesn't exist, create it
if not os.path.isdir(base_path):
  os.makedirs(base_path)

with app:

  UI.browser_timezone.render()

  # Render an invisible checkbox group to enable loading list of projects via API
  set_get_projects_api()

  with UI.separate_tab_warning.render():

    UI.separate_tab_link.render()

    gr.Button('Click here to open the UI', variant = 'primary' ).click( inputs = UI.separate_tab_link, outputs = None, fn = None,
      _js = "link => window.open(link, '_blank')"
    )
  
  with UI.main_window.render():

    render_sidebar()
    render_getting_started()
    render_workspace()

  on_app_load()

  app.launch( share = share_gradio, debug = debug_gradio )