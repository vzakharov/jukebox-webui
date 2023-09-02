import lib.ui.components.upsampling
from lib.upsampling.Upsampling import Upsampling

def monitor_upsampling_status():
    lib.ui.components.upsampling.upsampling_refresher.render().change(
      inputs = [ lib.ui.components.upsampling.upsampling_refresher, lib.ui.components.upsampling.upsampling_audio_refresher ],
      outputs = [ lib.ui.components.upsampling.upsampling_refresher, lib.ui.components.upsampling.upsampling_status, lib.ui.components.upsampling.upsampling_audio_refresher ],
      fn = lambda refresher, audio_refresher: {
        lib.ui.components.upsampling.upsampling_status: Upsampling.status_markdown,
        lib.ui.components.upsampling.upsampling_refresher: refresher + 1,
        lib.ui.components.upsampling.upsampling_audio_refresher: audio_refresher + 1 if Upsampling.should_refresh_audio else audio_refresher
      },
      _js = """
        async ( ...args ) => {
          await new Promise( resolve => setTimeout( resolve, 10000 ) )
          console.log( 'Checking upsampling status...' )
          return args
        }
      """
    )