import random

import gradio as gr

from lib.navigation.get_sample_filename import get_sample_filename
from lib.navigation.get_sibling_samples import get_sibling_samples
from lib.navigation.refresh_siblings import refresh_siblings
from lib.ui.components.workspace.first_generation import \
    render_first_generation
from lib.ui.components.workspace.sample_box.sample_box import render_sample_box
from lib.ui.components.workspace.sample_box.upsampling.init_args import \
    upsample_button_click_args
from lib.ui.components.workspace.sample_tree import render_sample_tree
from lib.ui.components.workspace.tabs.panic.panic import render_panic_tab
from lib.ui.components.workspace.tabs.prime import render_prime_tab
from lib.ui.js.update_url import update_url_js
from lib.ui.preview import default_preview_args, preview_inputs
from lib.ui.UI import UI
from lib.upsampling.start_upsampling import start_upsampling
from lib.upsampling.Upsampling import Upsampling

def render_workspace(app):
  with UI.workspace_column.render():
    with gr.Tab('Workspace'):
      with gr.Column():

        render_first_generation()
        render_sample_tree()

        UI.picked_sample.render()

        UI.sample_tree.change(
          inputs = [ UI.project_name, UI.sample_tree ],
          outputs = UI.picked_sample,
          fn = refresh_siblings,
          api_name = 'get-siblings'
        )

        # Virtual input & handler to create an API method for get_sample_filename
        gr.Textbox(visible=False).change(
          inputs = preview_inputs,
          outputs = gr.Textbox(visible=False),
          fn = get_sample_filename,
          api_name = 'get-sample-filename'
        )
    
        UI.picked_sample.change(
          **default_preview_args,
          api_name = 'get-sample',
          _js = update_url_js
        )

        # When the picked sample is updated, update all the others too (UI.sibling_chunks) by calling get_sample for each sibling
        UI.picked_sample_updated.render().change(
          inputs = [ *preview_inputs ],
          outputs = UI.sibling_chunks,
          fn = get_sibling_samples,
          api_name = 'get-sibling-samples',
        )

        UI.current_chunks.render()
        UI.sibling_chunks.render()

        UI.upsampled_lengths.render().change(
          inputs = UI.upsampled_lengths,
          outputs = None,
          fn = None,
          # Split by comma and turn into floats and add wavesurfer markers for each (first clear all the markers)
          _js = 'comma_separated => Ji.addUpsamplingMarkers( comma_separated.split(",").map( parseFloat ) )'
        )

        render_sample_box(app)

      UI.generation_progress.render()

    render_prime_tab()

    with gr.Tab('Upsample'):
      # Warning that this process is slow and can take up to 10 minutes for 1 second of audio
      with gr.Accordion('What is this?', open = False):
        gr.Markdown('''
          Upsampling is a process that creates higher-quality audio from your composition. It is done in two steps:
          - “Midsampling,” which considerably improves the quality of the audio, takes around 2 minutes per one second of audio.
          - “Upsampling,” which improves the quality some more, goes after midsampling and takes around 8 minutes per one second of audio.
          Thus, say, for a one-minute song, you will need to wait around 2 hours to have the midsampled version, and around 8 hours _more_ to have the upsampled version.
          You will be able to listen to the audio as it is being generated: Each “window” of upsampling takes ~6 minutes and will give you respectively ~2.7 and ~0.7 additional seconds of mid- or upsampled audio to listen to.
          ⚠️ WARNING: As the upsampling process uses a different model, which cannot be loaded together with the composition model due to memory constraints, **you will not be able to upsample and compose at the same time**. To go back to composing you will have to restart the Colab runtime or start a second Colab runtime and use them in parallel.
        ''')

      UI.sample_to_upsample.render()

      # Change the sample to upsample when a sample is picked
      UI.picked_sample.change(
        inputs = UI.picked_sample,
        outputs = UI.sample_to_upsample,
        fn = lambda x: x,
      )

      with gr.Accordion('Genres for upsampling (optional)', open = False):
        with gr.Accordion('What is this?', open = False):
          gr.Markdown('''
            The tool will generate three upsamplings of the selected sample, which will then be panned to the left, center, and right, respectively. Choosing different genres for each of the three upsamplings will result in a more diverse sound between them, thus enhancing the (pseudo-)stereo effect. 
            A good starting point is to have a genre that emphasizes vocals (e.g. `Pop`) for the center channel, and two similar but different genres for the left and right channels (e.g. `Rock` and `Metal`).
            If you don’t want to use this feature, simply select the same genre for all three upsamplings.
          ''')

        with gr.Row():
          for input in [ UI.genre_for_upsampling_left_channel, UI.genre_for_upsampling_center_channel, UI.genre_for_upsampling_right_channel ]:
            input.render()

        UI.kill_runtime_once_done.render()

      # If upsampling is running, enable the upsampling_refresher -- a "virtual" input that, when changed, will update the upsampling_status_markdown
      # It will do so after waiting for 10 seconds (using js). After finishing, it will update itself again, causing the process to repeat.
      UI.upsampling_refresher.render().change(
        inputs = [ UI.upsampling_refresher, UI.upsampling_audio_refresher ],
        outputs = [ UI.upsampling_refresher, UI.upsampling_status, UI.upsampling_audio_refresher ],
        fn = lambda refresher, audio_refresher: {
          UI.upsampling_status: Upsampling.status_markdown,
          UI.upsampling_refresher: refresher + 1,
          UI.upsampling_audio_refresher: audio_refresher + 1 if Upsampling.should_refresh_audio else audio_refresher
        },
        _js = """
          async ( ...args ) => {
            await new Promise( resolve => setTimeout( resolve, 10000 ) )
            console.log( 'Checking upsampling status...' )
            return args
          }
        """
      )

      UI.upsample_button.render().click( **upsample_button_click_args )

      # During app load, set upsampling_running and upsampling_stopping according to Upsampling.running
      app.load(
        inputs = None,
        outputs = UI.upsampling_running,
        fn = lambda: Upsampling.running,
      )

      UI.upsampling_triggered_by_button.render()

      # When upsampling_running changes via the button, run the upsampling process
      UI.upsampling_running.render().change(
        inputs = [
          UI.upsampling_triggered_by_button,
          UI.project_name, UI.sample_to_upsample, UI.artist, UI.lyrics,
          UI.genre_for_upsampling_left_channel, UI.genre_for_upsampling_center_channel, UI.genre_for_upsampling_right_channel,
          UI.kill_runtime_once_done
        ],
        outputs = None,
        fn = lambda triggered_by_button, *args: start_upsampling( *args ) if triggered_by_button else None,
        api_name = 'toggle-upsampling',
        # Also go to the "Workspace" tab (because that's where we'll display the upsampling status) via the Ji.clickTabWithText helper method in js
        _js = """
          async ( ...args ) => {
            console.log( 'Upsampling toggled, args:', args )
            if ( args[0] ) {
              Ji.clickTabWithText( 'Workspace' )
              return args
            } else {
              throw new Error('Upsampling not triggered by button')
            }
          }
        """
      )

      # When it changes regardless of the session, e.g. also at page refresh, update the various relevant UI elements, start the refresher, etc.
      UI.upsampling_running.change(
        inputs = None,
        outputs = [ UI.upsampling_status, UI.upsample_button, UI.continue_upsampling_button, UI.upsampling_refresher ],
        fn = lambda: {
          UI.upsampling_status: 'Upsampling in progress...',
          UI.upsample_button: gr.update(
            value = 'Stop upsampling',
            variant = 'secondary',
          ),
          UI.continue_upsampling_button: gr.update(
            value = 'Stop upsampling',
          ),
          # Random refresher value (int) to trigger the refresher
          UI.upsampling_refresher: random.randint( 0, 1000000 ),
          # # Hide the compose row
          # UI.compose_row: HIDE,
        }
      )

    render_panic_tab()