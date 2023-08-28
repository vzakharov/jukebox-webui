from lib.ui.UI import UI
from lib.ui.components.workspace.sample_box.upsampling.init_args import upsample_button_click_args
from lib.ui.components.workspace.sample_box.upsampling.manipulation import render_manipulation_column
from lib.ui.components.workspace.sample_box.upsampling.refresher import render_refresher
from lib.upsampling.show_or_hide_continue_upsampling import show_or_hide_continue_upsampling
from lib.upsampling.show_or_hide_upsampling_elements import show_or_hide_upsampling_elements
from lib.ui.preview import default_preview_args

import gradio as gr

def render_upsampling_accordion():

  with UI.upsampling_accordion.render():
    with gr.Row():
      
      with gr.Column():
        UI.upsampling_level.render().change(
          **default_preview_args,
        )

        show_or_hide_upsampling_elements_args = dict(
          inputs = [ UI.project_name, UI.picked_sample, UI.upsampling_running ],
          outputs = [ UI.upsampling_status, UI.upsampling_level ],
          fn = show_or_hide_upsampling_elements,
        )

        UI.picked_sample.change( **show_or_hide_upsampling_elements_args )
        UI.upsampling_running.change( **show_or_hide_upsampling_elements_args )

      render_manipulation_column()

    UI.picked_sample.change(
      inputs = [ UI.project_name, UI.picked_sample, UI.total_audio_length, UI.upsampling_running ],
      outputs = UI.continue_upsampling_button,
      fn = show_or_hide_continue_upsampling,
    )

    UI.continue_upsampling_button.render().click( **upsample_button_click_args )

    render_refresher(show_or_hide_upsampling_elements_args)

  UI.upsampling_status.render()
  
  return show_or_hide_upsampling_elements_args, upsample_button_click_args

