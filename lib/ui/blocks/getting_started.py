from UI.misc import getting_started_column

import gradio as gr

import urllib.request

def render_getting_started():
  with getting_started_column.render():
    # Load the getting started text from github (vzakharov/jukebox-webui/docs/getting-started.md) via urllib
    with urllib.request.urlopen('https://raw.githubusercontent.com/vzakharov/jukebox-webui/main/docs/getting-started.md') as f:
      getting_started_text = f.read().decode('utf-8')
      gr.Markdown(getting_started_text)