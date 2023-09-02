
import gradio as gr

import random

import UI.upsampling

def handle_upsampling_status_change():
  UI.upsampling.upsampling_running.change(
    inputs = None,
    outputs = [ UI.upsampling.upsampling_status, UI.upsampling.upsample_button, UI.upsampling.continue_upsampling_button, UI.upsampling.upsampling_refresher ],
    fn = lambda: {
      UI.upsampling.upsampling_status: 'Upsampling in progress...',
      UI.upsampling.upsample_button: gr.update(
        value = 'Stop upsampling',
        variant = 'secondary',
      ),
      UI.upsampling.continue_upsampling_button: gr.update(
        value = 'Stop upsampling',
      ),
      # Random refresher value (int) to trigger the refresher
      UI.upsampling.upsampling_refresher: random.randint( 0, 1000000 ),
      # # Hide the compose row
      # UI.compose_row: HIDE,
    }
  )