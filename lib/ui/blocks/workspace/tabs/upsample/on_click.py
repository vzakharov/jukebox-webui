import lib.ui.components.metas
import lib.ui.components.general
import lib.ui.components.upsampling
from lib.upsampling.start_upsampling import start_upsampling

def handle_upsampling_button_click():
  lib.ui.components.upsampling.upsampling_running.render().change(
    inputs = [
      lib.ui.components.upsampling.upsampling_triggered_by_button,
      lib.ui.components.general.project_name, lib.ui.components.upsampling.sample_to_upsample, lib.ui.components.metas.artist, lib.ui.components.metas.lyrics,
      lib.ui.components.upsampling.genre_for_upsampling_left_channel, lib.ui.components.upsampling.genre_for_upsampling_center_channel, lib.ui.components.upsampling.genre_for_upsampling_right_channel,
      lib.ui.components.upsampling.kill_runtime_once_done
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