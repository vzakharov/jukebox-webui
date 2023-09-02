import UI.upsampling
from lib.ui.js.confirm_upsampling import confirm_upsampling_js

import os
import subprocess

upsample_button_click_args = dict(
  inputs = UI.upsampling.upsampling_running,
  outputs = [ UI.upsampling.upsampling_running, UI.upsampling.upsampling_triggered_by_button ],
  fn = lambda was_running:
  # If was running (i.e. we're stopping), kill the runtime (after a warning) and show an alert saying to restart the runtime in Colab
    [
      print('Killing runtime...'),
      subprocess.run(['kill', '-9', str(os.getpid())]),
    ] if was_running else {
      UI.upsampling.upsampling_running: 1,
      UI.upsampling.upsampling_triggered_by_button: True,
    },
  _js = confirm_upsampling_js
)