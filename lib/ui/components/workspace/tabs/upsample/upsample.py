import gradio as gr

from lib.ui.components.workspace.sample_box.upsampling.init_args import \
    upsample_button_click_args
from lib.ui.UI import UI
from lib.upsampling.Upsampling import Upsampling

from .genres_for_upsampling import render_genres_for_upsampling
from .monitor import monitor_upsampling_status
from .on_change import handle_upsampling_status_change
from .on_click import handle_upsampling_button_click
from .what_is import what_is_upsampling_markdown

def render_upsample_tab(app):
    
  with gr.Tab('Upsample'):
    
    # Warning that this process is slow and can take up to 10 minutes for 1 second of audio
    with gr.Accordion('What is this?', open = False):
      gr.Markdown(what_is_upsampling_markdown)

    UI.sample_to_upsample.render()

    # Change the sample to upsample when a sample is picked
    UI.picked_sample.change(
      inputs = UI.picked_sample,
      outputs = UI.sample_to_upsample,
      fn = lambda x: x,
    )

    render_genres_for_upsampling()

    # If upsampling is running, enable the upsampling_refresher -- a "virtual" input that, when changed, will update the upsampling_status_markdown
    # It will do so after waiting for 10 seconds (using js). After finishing, it will update itself again, causing the process to repeat.
    monitor_upsampling_status()

    UI.upsample_button.render().click( **upsample_button_click_args )

    # During app load, set upsampling_running and upsampling_stopping according to Upsampling.running
    app.load(
      inputs = None,
      outputs = UI.upsampling_running,
      fn = lambda: Upsampling.running,
    )

    UI.upsampling_triggered_by_button.render()

    # When upsampling_running changes via the button, run the upsampling process
    handle_upsampling_button_click()

    # When it changes regardless of the session, e.g. also at page refresh, update the various relevant UI elements, start the refresher, etc.
    handle_upsampling_status_change()

