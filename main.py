import hashlib
import os
import sys
import urllib.request
from datetime import timedelta, timezone

import gradio as gr
from jukebox.sample import load_prompts

from lib.api import set_get_projects_api
from lib.cut import cut_zs
from lib.model.load import load_model
from lib.model.params import hps, set_hyperparams
from lib.ui.app_layout import app_layout
from lib.ui.on_load import on_load
from lib.ui.UI import UI
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

with app_layout() as app:

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
    render_workspace(app)

  # TODO: Don't forget to remove this line before publishing the app
  frontend_on_load_url = f'https://cdn.jsdelivr.net/gh/vzakharov/jukebox-webui@{GITHUB_SHA}/frontend-on-load.js'
  with urllib.request.urlopen(frontend_on_load_url) as response:
    frontend_on_load_js = response.read().decode('utf-8')

    try:
      old_frontend_on_load_md5 = frontend_on_load_md5
    except NameError:
      old_frontend_on_load_md5 = None

    frontend_on_load_md5 = hashlib.md5(frontend_on_load_js.encode('utf-8')).hexdigest()
    print(f'Loaded frontend-on-load.js from {response.geturl()}, md5: {frontend_on_load_md5}')

    if frontend_on_load_md5 != old_frontend_on_load_md5:
      print('(New version)')
    else:
      print('(Same version as during the previous run)')

    # print(frontend_on_load_js)

  app.load(
    on_load,
    inputs = [ gr.Textbox(visible=False), gr.Textbox(visible=False), gr.Textbox(visible=False) ],
    outputs = [ 
      UI.project_name, UI.routed_sample_id, UI.artist, UI.genre_dropdown, UI.getting_started_column, UI.separate_tab_warning, UI.separate_tab_link, UI.main_window,
      UI.genre_for_upsampling_left_channel, UI.genre_for_upsampling_center_channel, UI.genre_for_upsampling_right_channel
    ],
    api_name = 'initialize',
    _js = frontend_on_load_js,
    # _js = """
    # // (insert manually for debugging)
    # """,
  )

  # Also load browser's time zone offset on app load
  def set_browser_timezone(offset):
    global browser_timezone

    print('Browser time zone offset:', offset)
    browser_timezone = timezone(timedelta(minutes = -offset))
    print('Browser time zone:', browser_timezone)

  app.load(
    inputs = gr.Number( visible = False ),
    outputs = None,
    _js = '() => [ new Date().getTimezoneOffset() ]',
    fn = set_browser_timezone
  )

  app.launch( share = share_gradio, debug = debug_gradio )