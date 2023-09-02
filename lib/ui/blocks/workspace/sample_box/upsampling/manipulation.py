import lib.ui.components.upsampling
from lib.ui.preview import default_preview_args

import gradio as gr

def render_manipulation_column():
  with gr.Column() as upsampling_manipulation_column:
    # # Show the column only if an upsampled sample is selected and hide the compose row respectively (we can only compose with the original sample)
    # UI.upsampling_level.change(
    #   inputs = [ UI.upsampling_level, UI.upsampling_running ],
    #   outputs = [ upsampling_manipulation_column, UI.compose_row ],
    #   fn = lambda upsampling_level, upsampling_running: [
    #     gr.update( visible = upsampling_level != 'Raw' ),
    #     gr.update( visible = upsampling_level == 'Raw' and not upsampling_running ),
    #   ]
    # )
    with gr.Row():
      lib.ui.components.upsampling.upsample_rendering.render().change(
              **default_preview_args,
            )

      lib.ui.components.upsampling.combine_upsampling_levels.render().change(
              **default_preview_args,
            )

      lib.ui.components.upsampling.invert_center_channel.render().change(
              **default_preview_args,
            )