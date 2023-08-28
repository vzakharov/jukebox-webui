from lib.ui.UI import UI
from lib.ui.components.workspace.sample_box.compose_row import render_compose_row
from lib.ui.components.workspace.sample_box.advanced.advanced import render_advanced
from lib.ui.html.play_pause import play_pause_button
from lib.ui.components.workspace.sample_box.upsampling.state import render_upsampling_accordion
from lib.ui.preview import get_preview_args

import gradio as gr

def render_sample_box(app):
  with UI.sample_box.render():

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
      UI.audio_waveform,
      UI.audio_timeline
    ]:
      element.render()

    # Play/pause button, html-based
    gr.HTML(play_pause_button)

    render_compose_row()
    render_advanced(app)

  return upsample_button_click_args

