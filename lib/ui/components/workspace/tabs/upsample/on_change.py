import lib.ui.UI as UI

import gradio as gr

import random

def handle_upsampling_status_change():
  UI.upsampling_running.change(
    inputs = None,
    outputs = [ UI.upsampling_status, UI.upsample_button, UI.continue_upsampling_button, UI.upsampling_refresher ],
    fn = lambda: {
      UI.upsampling_status: 'Upsampling in progress...',
      UI.upsample_button: gr.update(
        value = 'Stop upsampling',
        variant = 'secondary',
      ),
      UI.continue_upsampling_button: gr.update(
        value = 'Stop upsampling',
      ),
      # Random refresher value (int) to trigger the refresher
      UI.upsampling_refresher: random.randint( 0, 1000000 ),
      # # Hide the compose row
      # UI.compose_row: HIDE,
    }
  )