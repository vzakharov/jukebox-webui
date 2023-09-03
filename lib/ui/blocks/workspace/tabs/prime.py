import gradio as gr

from lib.audio.to_sample import to_sample
from lib.audio.utils import trim_primed_audio
from lib.ui.elements.first import (first_generation_row, prime_timestamp,
                                   primed_audio)
from lib.ui.elements.general import project_name
from lib.ui.elements.navigation import sample_tree, show_leafs_only


def render_prime_tab():
  with gr.Tab('Prime'):
    primed_audio_source = gr.Radio(
      label = 'Audio source',
      choices = [ 'microphone', 'upload' ],
      value = 'microphone'
    )

    primed_audio.render()

    primed_audio_source.change(
      inputs = primed_audio_source,
      outputs = primed_audio,
      fn = lambda source: gr.update( source = source ),
    )

    sec_to_trim_primed_audio = gr.Number(
      label = 'Trim starting audio to ... seconds from the beginning',
    )

    sec_to_trim_primed_audio.submit(
      inputs = [ primed_audio, sec_to_trim_primed_audio ],
      outputs = primed_audio,
      fn = trim_primed_audio
    )

    prime_button = gr.Button(
      'Convert to sample',
      variant = 'primary'
    )

    prime_button.click(
      inputs = [ project_name, primed_audio, sec_to_trim_primed_audio, show_leafs_only ],
      outputs = [ sample_tree, prime_button, prime_timestamp, first_generation_row ], # UI.prime_timestamp is updated to the current time to force tab change
      fn = to_sample,
      api_name = 'convert-wav-to-sample'
    )

    prime_timestamp.render().change(
      inputs = prime_timestamp, outputs = None, fn = None,
      _js = """
        timestamp => {
          Ji.clickTabWithText('Workspace')
          return timestamp
        }
      """
      # Gosh, this is ugly.
    )