import gradio as gr

from lib.ui.adjust_max_samples import adjust_max_samples
from lib.ui.blocks.workspace.sample_box.advanced.cut_audio.render_cut_audio import \
    render_cut_audio
from lib.ui.preview import default_preview_args
from lib.ui.elements.preview import cut_audio_specs
from lib.ui.elements.project import total_audio_length

def render_manipulate_tab():
  with gr.Tab('Manipulate audio'):
    total_audio_length.render()

    # TODO: move this to another place
    adjust_max_samples()

    cut_audio_specs.render().submit(**default_preview_args)

    render_cut_audio()