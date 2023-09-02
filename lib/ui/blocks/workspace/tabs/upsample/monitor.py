from UI.upsampling import upsampling_refresher, upsampling_refresher, upsampling_audio_refresher, upsampling_refresher, upsampling_status, upsampling_audio_refresher, upsampling_status, upsampling_refresher, upsampling_audio_refresher
from lib.upsampling.Upsampling import Upsampling

def monitor_upsampling_status():
    upsampling_refresher.render().change(
      inputs = [ upsampling_refresher, upsampling_audio_refresher ],
      outputs = [ upsampling_refresher, upsampling_status, upsampling_audio_refresher ],
      fn = lambda refresher, audio_refresher: {
        upsampling_status: Upsampling.status_markdown,
        upsampling_refresher: refresher + 1,
        upsampling_audio_refresher: audio_refresher + 1 if Upsampling.should_refresh_audio else audio_refresher
      },
      _js = """
        async ( ...args ) => {
          await new Promise( resolve => setTimeout( resolve, 10000 ) )
          console.log( 'Checking upsampling status...' )
          return args
        }
      """
    )