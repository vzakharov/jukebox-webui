import lib.ui.components as UI
from lib.ui.preview import default_preview_args
from lib.upsampling.Upsampling import Upsampling

def render_refresher(show_or_hide_upsampling_elements_args):
  UI.upsampling_audio_refresher.render()

  def reset_audio_refresher():
    Upsampling.should_refresh_audio = False

  [
    UI.upsampling_audio_refresher.change( **action ) for action in [
      default_preview_args,
      show_or_hide_upsampling_elements_args,
      dict(
        inputs = None,
        outputs = None,
        # Reset Upsampling.should_refresh_audio to False
        fn = reset_audio_refresher
      )
    ]
  ]