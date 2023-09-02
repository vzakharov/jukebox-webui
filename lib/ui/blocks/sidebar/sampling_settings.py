from lib.lists import get_list
from lib.navigation.save_project import save_project

import gradio as gr

import random
from UI.project import generation_params, generation_discard_window, project_settings, project_settings
from UI.metas import artist, genre_dropdown, genre

from UI.general import settings_box, project_name

def render_sampling_settings():
  with settings_box.render():
    
    for component in generation_params:
      # For artist, also add a search button and a randomize button
      if component == artist:
        with gr.Row():
          component.render()

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

      elif component == genre:
        genre_dropdown.render().change(
          inputs = [ genre, genre_dropdown ],
          outputs = genre,
          # Add after a space, if not empty
          fn = lambda genre, genre_dropdown: ( genre + ' ' if genre else '' ) + genre_dropdown,
        )

        component.render()

      elif component == generation_discard_window:
        component.render()

        with gr.Accordion( 'What is this?', open = False ):
          gr.Markdown("""
            If your song is too long, the generation may take too much memory and crash. In this case, you can discard the first N seconds of the song for generation purposes (i.e. the model won’t take them into account when generating the rest of the song).
            If your song has lyrics, put '---' (with a new line before and after) at the point that is now the “beginning” of the song, so that the model doesn’t get confused by the now-irrelevant lyrics.
          """)

      else:
        component.render()

    for component in project_settings:
      # Whenever a project setting is changed, save all the settings to settings.yaml in the project folder
      inputs = [ project_name, *project_settings ]

      # Use the "blur" method if available, otherwise use "change"
      handler_name = 'blur' if hasattr(component, 'blur') else 'change'
      handler = getattr(component, handler_name)

      handler(
        inputs = inputs,
        outputs = None,
        fn = save_project,
      )