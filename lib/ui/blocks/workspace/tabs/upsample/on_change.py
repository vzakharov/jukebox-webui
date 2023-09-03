
import gradio as gr

import random

from lib.ui.elements.upsampling import upsampling_running, upsampling_status, upsample_button, continue_upsampling_button, upsampling_refresher, upsampling_status, upsample_button, continue_upsampling_button, upsampling_refresher

def handle_upsampling_status_change():
  upsampling_running.change(
    inputs = None,
    outputs = [ upsampling_status, upsample_button, continue_upsampling_button, upsampling_refresher ],
    fn = lambda: {
      upsampling_status: 'Upsampling in progress...',
      upsample_button: gr.update(
        value = 'Stop upsampling',
        variant = 'secondary',
      ),
      continue_upsampling_button: gr.update(
        value = 'Stop upsampling',
      ),
      # Random refresher value (int) to trigger the refresher
      upsampling_refresher: random.randint( 0, 1000000 ),
      # # Hide the compose row
      # UI.compose_row: HIDE,
    }
  )