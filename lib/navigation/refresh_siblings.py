from .utils import get_siblings
from lib.ui.UI import UI
from lib.ui.utils import HIDE

import gradio as gr

def refresh_siblings(project_name, sample_id):

  if not sample_id:
    return {
      UI.picked_sample: HIDE
    }

  # print(f'Getting siblings for {sample_id}...')
  siblings = get_siblings(project_name, sample_id)
  # print(f'Siblings for {sample_id}: {siblings}')
  return gr.update(
    choices = siblings,
    value = sample_id,
    visible = len(siblings) > 1
  )