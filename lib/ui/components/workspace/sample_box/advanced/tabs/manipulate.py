from lib.ui.UI import UI
from lib.ui.adjust_max_samples import adjust_max_samples
from lib.ui.components.workspace.sample_box.advanced.cut_audio.render_cut_audio import render_cut_audio
from lib.ui.preview import default_preview_args

import gradio as gr

def render_manipulate_tab():
  with gr.Tab('Manipulate audio'):
    UI.total_audio_length.render()

    # TODO: move this to another place
    adjust_max_samples()

    UI.cut_audio_specs.render().submit(**default_preview_args)

    render_cut_audio()