from UI.metas import artist, lyrics
from UI.general import project_name
from UI.upsampling import upsampling_running, upsampling_triggered_by_button, sample_to_upsample, genre_left_channel, genre_center_channel, genre_right_channel, kill_runtime_once_done
from lib.upsampling.start_upsampling import start_upsampling

def handle_upsampling_button_click():
  upsampling_running.render().change(
    inputs = [
      upsampling_triggered_by_button,
      project_name, sample_to_upsample, artist, lyrics,
      genre_left_channel, genre_center_channel, genre_right_channel,
      kill_runtime_once_done
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