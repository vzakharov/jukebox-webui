import UI.metas
import UI.general
import UI.upsampling
from lib.upsampling.start_upsampling import start_upsampling

def handle_upsampling_button_click():
  UI.upsampling.upsampling_running.render().change(
    inputs = [
      UI.upsampling.upsampling_triggered_by_button,
      UI.general.project_name, UI.upsampling.sample_to_upsample, UI.metas.artist, UI.metas.lyrics,
      UI.upsampling.genre_left_channel, UI.upsampling.genre_center_channel, UI.upsampling.genre_right_channel,
      UI.upsampling.kill_runtime_once_done
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