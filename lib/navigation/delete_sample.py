import lib.ui.components.navigation
from .utils import get_siblings
from params import base_path

import gradio as gr

import os

def delete_sample(project_name, sample_id, confirm):

  if not confirm:
    return {}

  # New child sample is the one that goes after the deleted sample
  siblings = get_siblings(project_name, sample_id)
  current_index = siblings.index(sample_id)
  new_sibling_to_use = siblings[ current_index + 1 ] if current_index < len(siblings) - 1 else siblings[ current_index - 1 ]

  # Remove the to-be-deleted sample from the list of child samples
  siblings.remove(sample_id)

  # Delete the sample
  filename = f'{base_path}/{project_name}/{sample_id}'

  for extension in [ '.z', '.wav' ]:
    if os.path.isfile(f'{filename}{extension}'):
      os.remove(f'{filename}{extension}')
      print(f'Deleted {filename}{extension}')
    else:
      print(f'No {filename}{extension} found')
  return {
    lib.ui.components.navigation.picked_sample: gr.update(
      choices = siblings,
      value = new_sibling_to_use,
    ),
    lib.ui.components.navigation.sample_box: gr.update(
      visible = len(siblings) > 0
    ),
  }