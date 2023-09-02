import UI.upsampling
from lib.upsampling.Upsampling import Upsampling

def monitor_upsampling_status():
    UI.upsampling.upsampling_refresher.render().change(
      inputs = [ UI.upsampling.upsampling_refresher, UI.upsampling.upsampling_audio_refresher ],
      outputs = [ UI.upsampling.upsampling_refresher, UI.upsampling.upsampling_status, UI.upsampling.upsampling_audio_refresher ],
      fn = lambda refresher, audio_refresher: {
        UI.upsampling.upsampling_status: Upsampling.status_markdown,
        UI.upsampling.upsampling_refresher: refresher + 1,
        UI.upsampling.upsampling_audio_refresher: audio_refresher + 1 if Upsampling.should_refresh_audio else audio_refresher
      },
      _js = """
        async ( ...args ) => {
          await new Promise( resolve => setTimeout( resolve, 10000 ) )
          console.log( 'Checking upsampling status...' )
          return args
        }
      """
    )