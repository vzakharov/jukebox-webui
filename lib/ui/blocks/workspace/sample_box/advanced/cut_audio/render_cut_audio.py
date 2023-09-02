from lib.audio.cut import cut_audio
import lib.ui.components.general
import lib.ui.components.navigation
import lib.ui.components.preview
from .how_to import how_to_cut_audio_markdown
from lib.ui.preview import default_preview_args

import gradio as gr

def render_cut_audio():
  with gr.Row():
    lib.ui.components.preview.cut_audio_preview_button.render().click(**default_preview_args)

    # Make the cut out buttons visible or not depending on whether the cut out value is 0
    lib.ui.components.preview.cut_audio_specs.change(
      inputs = lib.ui.components.preview.cut_audio_specs,
      outputs = [ lib.ui.components.preview.cut_audio_preview_button, lib.ui.components.preview.cut_audio_apply_button ],
      fn = lambda cut_audio_specs: [
        gr.update( visible = cut_audio_specs != '' ) for _ in range(3)
      ]
    )

    lib.ui.components.preview.cut_audio_apply_button.render().click(
      inputs = [ lib.ui.components.general.project_name, lib.ui.components.navigation.picked_sample, lib.ui.components.preview.cut_audio_specs ],
      outputs = lib.ui.components.preview.cut_audio_specs,
      fn = cut_audio,
      api_name = 'cut-audio',
    )

  with gr.Accordion('How does it work?', open = False):
    gr.Markdown(how_to_cut_audio_markdown)

  lib.ui.components.preview.preview_just_the_last_n_sec.render().blur(**default_preview_args)