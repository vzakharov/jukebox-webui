from lib.ui.elements.upsampling import upsampling_running, upsampling_running, upsampling_triggered_by_button, upsampling_running, upsampling_triggered_by_button
from lib.ui.js.confirm_upsampling import confirm_upsampling_js

import os
import subprocess

upsample_button_click_args = dict(
  inputs = upsampling_running,
  outputs = [ upsampling_running, upsampling_triggered_by_button ],
  fn = lambda was_running:
  # If was running (i.e. we're stopping), kill the runtime (after a warning) and show an alert saying to restart the runtime in Colab
    [
      print('Killing runtime...'),
      subprocess.run(['kill', '-9', str(os.getpid())]),
    ] if was_running else {
      upsampling_running: 1,
      upsampling_triggered_by_button: True,
    },
  _js = confirm_upsampling_js
)