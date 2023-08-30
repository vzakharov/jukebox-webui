from lib.ui.UI import UI
from lib.upsampling.Upsampling import Upsampling


def monitor_upsampling_status():
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