
import gradio as gr

import random

import lib.ui.components.upsampling

def handle_upsampling_status_change():
  lib.ui.components.upsampling.upsampling_running.change(
    inputs = None,
    outputs = [ lib.ui.components.upsampling.upsampling_status, lib.ui.components.upsampling.upsample_button, lib.ui.components.upsampling.continue_upsampling_button, lib.ui.components.upsampling.upsampling_refresher ],
    fn = lambda: {
      lib.ui.components.upsampling.upsampling_status: 'Upsampling in progress...',
      lib.ui.components.upsampling.upsample_button: gr.update(
        value = 'Stop upsampling',
        variant = 'secondary',
      ),
      lib.ui.components.upsampling.continue_upsampling_button: gr.update(
        value = 'Stop upsampling',
      ),
      # Random refresher value (int) to trigger the refresher
      lib.ui.components.upsampling.upsampling_refresher: random.randint( 0, 1000000 ),
      # # Hide the compose row
      # UI.compose_row: HIDE,
    }
  )