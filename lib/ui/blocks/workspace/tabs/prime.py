import gradio as gr

from lib.audio.convert_audio_to_sample import convert_audio_to_sample
from lib.audio.utils import trim_primed_audio
import lib.ui.components.first
import lib.ui.components.general
import lib.ui.components.navigation

def render_prime_tab():
  with gr.Tab('Prime'):
    primed_audio_source = gr.Radio(
      label = 'Audio source',
      choices = [ 'microphone', 'upload' ],
      value = 'microphone'
    )

    lib.ui.components.first.primed_audio.render()

    primed_audio_source.change(
      inputs = primed_audio_source,
      outputs = lib.ui.components.first.primed_audio,
      fn = lambda source: gr.update( source = source ),
    )

    sec_to_trim_primed_audio = gr.Number(
      label = 'Trim starting audio to ... seconds from the beginning',
    )

    sec_to_trim_primed_audio.submit(
      inputs = [ lib.ui.components.first.primed_audio, sec_to_trim_primed_audio ],
      outputs = lib.ui.components.first.primed_audio,
      fn = trim_primed_audio
    )

    prime_button = gr.Button(
      'Convert to sample',
      variant = 'primary'
    )

    prime_button.click(
      inputs = [ lib.ui.components.general.project_name, lib.ui.components.first.primed_audio, sec_to_trim_primed_audio, lib.ui.components.navigation.show_leafs_only ],
      outputs = [ lib.ui.components.navigation.sample_tree, prime_button, lib.ui.components.first.prime_timestamp, lib.ui.components.first.first_generation_row ], # UI.prime_timestamp is updated to the current time to force tab change
      fn = convert_audio_to_sample,
      api_name = 'convert-wav-to-sample'
    )

    lib.ui.components.first.prime_timestamp.render().change(
      inputs = lib.ui.components.first.prime_timestamp, outputs = None, fn = None,
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