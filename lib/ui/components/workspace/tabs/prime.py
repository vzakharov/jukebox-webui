from lib.audio.convert_audio_to_sample import convert_audio_to_sample
from lib.ui.UI import UI

import gradio as gr

from lib.audio.utils import trim_primed_audio

def render_prime_tab():
  with gr.Tab('Prime'):
    primed_audio_source = gr.Radio(
      label = 'Audio source',
      choices = [ 'microphone', 'upload' ],
      value = 'microphone'
    )

    UI.primed_audio.render()

    primed_audio_source.change(
      inputs = primed_audio_source,
      outputs = UI.primed_audio,
      fn = lambda source: gr.update( source = source ),
    )

    sec_to_trim_primed_audio = gr.Number(
      label = 'Trim starting audio to ... seconds from the beginning',
    )

    sec_to_trim_primed_audio.submit(
      inputs = [ UI.primed_audio, sec_to_trim_primed_audio ],
      outputs = UI.primed_audio,
      fn = trim_primed_audio
    )

    prime_button = gr.Button(
      'Convert to sample',
      variant = 'primary'
    )

    prime_button.click(
      inputs = [ UI.project_name, UI.primed_audio, sec_to_trim_primed_audio, UI.show_leafs_only ],
      outputs = [ UI.sample_tree, prime_button, UI.prime_timestamp, UI.first_generation_row ], # UI.prime_timestamp is updated to the current time to force tab change
      fn = convert_audio_to_sample,
      api_name = 'convert-wav-to-sample'
    )

    UI.prime_timestamp.render().change(
      inputs = UI.prime_timestamp, outputs = None, fn = None,
      _js =
        # Find a button inside a div inside another div with class 'tabs', the button having 'Workspace' as text, and click it -- all this in the shadow DOM.
        # Gosh, this is ugly.
        """
          timestamp => {
            console.log(`Timestamp changed to ${timestamp}; clicking the 'Workspace' tab`)
            Ji.clickTabWithText('Workspace')
            return timestamp
          }
        """
    )