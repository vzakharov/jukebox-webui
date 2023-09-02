from lib.audio.cut import cut_audio
import UI.general
import UI.navigation
import UI.preview
from .how_to import how_to_cut_audio_markdown
from lib.ui.preview import default_preview_args

import gradio as gr

def render_cut_audio():
  with gr.Row():
    UI.preview.cut_audio_preview_button.render().click(**default_preview_args)

    # Make the cut out buttons visible or not depending on whether the cut out value is 0
    UI.preview.cut_audio_specs.change(
      inputs = UI.preview.cut_audio_specs,
      outputs = [ UI.preview.cut_audio_preview_button, UI.preview.cut_audio_apply_button ],
      fn = lambda cut_audio_specs: [
        gr.update( visible = cut_audio_specs != '' ) for _ in range(3)
      ]
    )

    UI.preview.cut_audio_apply_button.render().click(
      inputs = [ UI.general.project_name, UI.navigation.picked_sample, UI.preview.cut_audio_specs ],
      outputs = UI.preview.cut_audio_specs,
      fn = cut_audio,
      api_name = 'cut-audio',
    )

  with gr.Accordion('How does it work?', open = False):
    gr.Markdown(how_to_cut_audio_markdown)

  UI.preview.just_the_last_n_sec.render().blur(**default_preview_args)