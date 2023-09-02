# Only show the upsampling elements if there are upsampled versions of the picked sample
from lib.navigation.utils import get_zs
from UI.upsampling import UPSAMPLING_LEVEL_NAMES, upsampling_status, upsampling_level
from .utils import get_levels

import gradio as gr

def show_or_hide_upsampling_elements(project_name, sample_id, upsampling_running):
  levels = get_levels(get_zs(project_name, sample_id))
  # print(f'Levels: {levels}')

  available_level_names = UPSAMPLING_LEVEL_NAMES[:len(levels)]
  print(f'Available level names: {available_level_names}')

  return {
    # UI.upsampling_accordion: gr.update(
    #   visible = len(levels) > 1 or upsampling_running,
    # ),
    # (removing the accordion for now)
    upsampling_status: gr.update(
      visible = upsampling_running,
    ),
    upsampling_level: gr.update(
      choices = available_level_names,
      # Choose the highest available level
      value = available_level_names[-1],
    )
  }