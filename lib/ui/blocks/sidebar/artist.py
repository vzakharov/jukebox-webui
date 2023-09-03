import random

import gradio as gr

from lib.lists import get_list
from lib.ui.elements.metas import artist


def filter_artists(filter):
  artists = get_list('artist')

  if filter:
    artists = [ artist for artist in artists if filter.lower() in artist.lower() ]
    artist = artists[0]
  else:
    # random artist
    artist = random.choice(artists)

  return gr.update(
    choices = artists,
    value = artist
  )


def render_artist():
  with gr.Row():
    artist.render()

    artist_filter = gr.Textbox(
      label = '🔍',
      placeholder = 'Empty for 🎲',
    )

    artist_filter.submit(
      inputs = artist_filter,
      outputs = artist,
      fn = filter_artists,
      api_name = 'filter-artists'
    )