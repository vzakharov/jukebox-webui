import UI.general
import UI.navigation
import UI.preview
import UI.upsampling
from .init_args import upsample_button_click_args
from .manipulation import render_manipulation_column
from .refresher import render_refresher
from lib.upsampling.show_or_hide_continue_upsampling import show_or_hide_continue_upsampling
from lib.upsampling.show_or_hide_upsampling_elements import show_or_hide_upsampling_elements
from lib.ui.preview import default_preview_args

import gradio as gr

def render_upsampling_accordion():

  with UI.upsampling.upsampling_accordion.render():
    with gr.Row():
      
      with gr.Column():
        UI.upsampling.upsampling_level.render().change(
          **default_preview_args,
        )

        show_or_hide_upsampling_elements_args = dict(
          inputs = [ UI.general.project_name, UI.navigation.picked_sample, UI.upsampling.upsampling_running ],
          outputs = [ UI.upsampling.upsampling_status, UI.upsampling.upsampling_level ],
          fn = show_or_hide_upsampling_elements,
        )

        UI.navigation.picked_sample.change( **show_or_hide_upsampling_elements_args )
        UI.upsampling.upsampling_running.change( **show_or_hide_upsampling_elements_args )

      render_manipulation_column()

    UI.navigation.picked_sample.change(
      inputs = [ UI.general.project_name, UI.navigation.picked_sample, UI.project.total_audio_length, UI.upsampling.upsampling_running ],
      outputs = UI.upsampling.continue_upsampling_button,
      fn = show_or_hide_continue_upsampling,
    )

    UI.upsampling.continue_upsampling_button.render().click( **upsample_button_click_args )

    render_refresher(show_or_hide_upsampling_elements_args)

  UI.upsampling.upsampling_status.render()
  
  return show_or_hide_upsampling_elements_args, upsample_button_click_args

