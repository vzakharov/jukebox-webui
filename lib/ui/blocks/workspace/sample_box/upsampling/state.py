import lib.ui.components.general
import lib.ui.components.navigation
import lib.ui.components.preview
import lib.ui.components.upsampling
from .init_args import upsample_button_click_args
from .manipulation import render_manipulation_column
from .refresher import render_refresher
from lib.upsampling.show_or_hide_continue_upsampling import show_or_hide_continue_upsampling
from lib.upsampling.show_or_hide_upsampling_elements import show_or_hide_upsampling_elements
from lib.ui.preview import default_preview_args

import gradio as gr

def render_upsampling_accordion():

  with lib.ui.components.upsampling.upsampling_accordion.render():
    with gr.Row():
      
      with gr.Column():
        lib.ui.components.upsampling.upsampling_level.render().change(
          **default_preview_args,
        )

        show_or_hide_upsampling_elements_args = dict(
          inputs = [ lib.ui.components.general.project_name, lib.ui.components.navigation.picked_sample, lib.ui.components.upsampling.upsampling_running ],
          outputs = [ lib.ui.components.upsampling.upsampling_status, lib.ui.components.upsampling.upsampling_level ],
          fn = show_or_hide_upsampling_elements,
        )

        lib.ui.components.navigation.picked_sample.change( **show_or_hide_upsampling_elements_args )
        lib.ui.components.upsampling.upsampling_running.change( **show_or_hide_upsampling_elements_args )

      render_manipulation_column()

    lib.ui.components.navigation.picked_sample.change(
      inputs = [ lib.ui.components.general.project_name, lib.ui.components.navigation.picked_sample, lib.ui.components.preview.total_audio_length, lib.ui.components.upsampling.upsampling_running ],
      outputs = lib.ui.components.upsampling.continue_upsampling_button,
      fn = show_or_hide_continue_upsampling,
    )

    lib.ui.components.upsampling.continue_upsampling_button.render().click( **upsample_button_click_args )

    render_refresher(show_or_hide_upsampling_elements_args)

  lib.ui.components.upsampling.upsampling_status.render()
  
  return show_or_hide_upsampling_elements_args, upsample_button_click_args

