from lib.ui.adjust_max_samples import adjust_max_samples
from lib.ui.blocks.workspace.sample_box.advanced.cut_audio.render_cut_audio import render_cut_audio
import UI.preview
from lib.ui.preview import default_preview_args

import gradio as gr

def render_manipulate_tab():
  with gr.Tab('Manipulate audio'):
    UI.project.total_audio_length.render()

    # TODO: move this to another place
    adjust_max_samples()

    UI.preview.cut_audio_specs.render().submit(**default_preview_args)

    render_cut_audio()