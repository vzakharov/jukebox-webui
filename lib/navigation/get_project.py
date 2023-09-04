import gradio as gr

from lib.ui.elements.first import first_generation_row
from lib.ui.elements.general import create_project_box, settings_box
from lib.ui.elements.main import workspace_column
from lib.ui.elements.metas import artist, genre, lyrics
from lib.ui.elements.navigation import sample_tree, sample_tree_row
from lib.ui.elements.project import generation_length, n_samples, temperature
from lib.ui.elements.sample import sample_box
from lib.ui.elements.upsampling import (genre_center_channel,
                                        genre_left_channel,
                                        genre_right_channel)
from params import base_path

from .sync_settings import sync_settings
from .utils import is_new, loaded_settings


def get_project(project_name, routed_sample_id):

  global base_path, loaded_settings
  print(f'Getting project {project_name} with routed sample id {routed_sample_id}')

  is_this_new = is_new(project_name)

  # Start with default values for project settings
  settings_out_dict = {
    artist: 'Unknown',
    genre: 'Unknown',
    lyrics: '',
    generation_length: 1,
    temperature: 0.98,
    n_samples: 2,
    sample_tree: None,
    genre_left_channel: 'Unknown',
    genre_center_channel: 'Unknown',
    genre_right_channel: 'Unknown',
  }

  samples = []
  sample = None

  # If not new, load the settings from settings.yaml in the project folder, if it exists
  if not is_this_new:
    samples, sample = sync_settings(project_name, routed_sample_id, settings_out_dict)

  return {
    create_project_box: gr.update( visible = is_this_new ),
    settings_box: gr.update( visible = not is_this_new ),
    workspace_column: gr.update( visible = not is_this_new  ),
    sample_box: gr.update( visible = sample is not None ),
    first_generation_row: gr.update( visible = len(samples) == 0 ),
    sample_tree_row: gr.update( visible = len(samples) > 0 ),
    **settings_out_dict
  }