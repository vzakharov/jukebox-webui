from lib.audio.cut import cut_audio
from lib.ui.elements.general import project_name
from lib.ui.elements.navigation import picked_sample
from lib.ui.elements.preview import cut_audio_preview_button, cut_audio_specs, cut_audio_specs, cut_audio_preview_button, cut_audio_apply_button, cut_audio_apply_button, cut_audio_specs, cut_audio_specs, just_the_last_n_sec
from .how_to import how_to_cut_audio_markdown
from lib.ui.preview import default_preview_args

import gradio as gr

def render_cut_audio():
  with gr.Row():
    cut_audio_preview_button.render().click(**default_preview_args)

    # Make the cut out buttons visible or not depending on whether the cut out value is 0
    cut_audio_specs.change(
      inputs = cut_audio_specs,
      outputs = [ cut_audio_preview_button, cut_audio_apply_button ],
      fn = lambda cut_audio_specs: [
        gr.update( visible = cut_audio_specs != '' ) for _ in range(3)
      ]
    )

    cut_audio_apply_button.render().click(
      inputs = [ project_name, picked_sample, cut_audio_specs ],
      outputs = cut_audio_specs,
      fn = cut_audio,
      api_name = 'cut-audio',
    )

  with gr.Accordion('How does it work?', open = False):
    gr.Markdown(how_to_cut_audio_markdown)

  just_the_last_n_sec.render().blur(**default_preview_args)