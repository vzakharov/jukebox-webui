# Show the continue upsampling markdown only if the current level's length in tokens is less than the total audio length
# Also update the upsampling button to say "Continue upsampling" instead of "Upsample"
from lib.navigation.zs import get_zs
from .utils import get_levels
from lib.utils import tokens_to_seconds

import gradio as gr

def show_or_hide_continue_upsampling(project_name, sample_id, total_audio_length, upsampling_running):
  if not upsampling_running:
    zs = get_zs(project_name, sample_id)
    levels = get_levels(zs)
        # print(f'Levels: {levels}, z: {zs}')
        # We'll show if there's no level 0 in levels or if the length of level 0 (in seconds) is less than the length of level 2 (in seconds)
    must_show = 0 not in levels or tokens_to_seconds(len(zs[0]), 0) < tokens_to_seconds(len(zs[2]), 2)
        # print(f'Must show: {must_show}')

  else:
    must_show = True

  return gr.update( visible = must_show )