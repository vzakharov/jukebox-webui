import lib.ui.components as UI
from lib.upsampling.start_upsampling import start_upsampling

def handle_upsampling_button_click():
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