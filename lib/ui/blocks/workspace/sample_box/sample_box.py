import gradio as gr

from lib.app import app
from lib.ui.elements.audio import audio_waveform, audio_timeline
from lib.ui.elements.navigation import sample_box
from lib.ui.html.play_pause import play_pause_button
from lib.ui.preview import get_preview_args

from .advanced.advanced import render_advanced
from .compose_row import render_compose_row
from .upsampling.state import render_upsampling_accordion

def render_sample_box():
  with sample_box.render():

    show_or_hide_upsampling_elements_args, upsample_button_click_args = render_upsampling_accordion()

    # Refresh button
    internal_refresh_button = gr.Button('🔃', elem_id = 'internal-refresh-button', visible=False)

    internal_refresh_button.click(
      **get_preview_args(force_reload = True),
    )

    internal_refresh_button.click(
      **show_or_hide_upsampling_elements_args,
    )

    for element in [
      audio_waveform,
      audio_timeline
    ]:
      element.render()

    # Play/pause button, html-based
    gr.HTML(play_pause_button)

    render_compose_row()
    render_advanced()

  return upsample_button_click_args

