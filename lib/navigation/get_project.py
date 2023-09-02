import os

import gradio as gr
import yaml

from lib.model.params import hps
from .utils import inputs_by_name
from lib.ui.elements.first import first_generation_row
from lib.ui.elements.general import create_project_box, settings_box
from lib.ui.elements.main import workspace_column
from lib.ui.elements.metas import artist, genre, lyrics
from lib.ui.elements.misc import getting_started_column
from lib.ui.elements.navigation import (sample_tree, sample_tree_row,
                                        show_leafs_only)
from lib.ui.elements.project import (generation_length, n_samples,
                                     project_settings, temperature)
from lib.ui.elements.sample import sample_box
from lib.ui.elements.upsampling import (genre_center_channel,
                                        genre_left_channel,
                                        genre_right_channel)
from params import base_path

from .get_samples import get_samples
from .utils import is_new, loaded_settings

def get_project(project_name, routed_sample_id):

  global base_path, loaded_settings

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

    print(f'Loading settings for {project_name}...')

    project_path = f'{base_path}/{project_name}'
    hps.name = project_path
    settings_path = f'{project_path}/{project_name}.yaml'
    if os.path.isfile(settings_path):
      with open(settings_path, 'r') as f:
        loaded_settings = yaml.load(f, Loader=yaml.FullLoader)
        print(f'Loaded settings for {project_name}: {loaded_settings}')

        # Go through all the settings and set the value for settings_out_dict where the key is the element itself
        for key, value in loaded_settings.items():
          if key in inputs_by_name and inputs_by_name[key] in project_settings:

            input = inputs_by_name[key]

            # If the value is an integer (i) but the element is an instance of gr.components.Radio or gr.components.Dropdown, take the i-th item from the choices
            if isinstance(value, int) and isinstance(input, (gr.components.Radio, gr.components.Dropdown)):
              print(f'Converting {key} value {value} to {input.choices[value]}')
              value = input.choices[value]

            # settings_out_dict[getattr(UI, key)] = value
            settings_out_dict[input] = value

          else:
            print(f'Warning: {key} is not a valid project setting')

    # Write the last project name to settings.yaml
    with open(f'{base_path}/settings.yaml', 'w') as f:
      print(f'Saving {project_name} as last project...')
      yaml.dump({'last_project': project_name}, f)
      print('Saved to settings.yaml')

    settings_out_dict[ getting_started_column ] = gr.update(
      visible = False
    )

    samples = get_samples(project_name, settings_out_dict[ show_leafs_only ] if show_leafs_only in settings_out_dict else False)

    sample = routed_sample_id or (
      (
        settings_out_dict[ sample_tree ] or samples[0]
      ) if len(samples) > 0 else None
    )

    settings_out_dict[ sample_tree ] = gr.update(
      choices = samples,
      value = sample
    )

  return {
    create_project_box: gr.update( visible = is_this_new ),
    settings_box: gr.update( visible = not is_this_new ),
    workspace_column: gr.update( visible = not is_this_new  ),
    sample_box: gr.update( visible = sample is not None ),
    first_generation_row: gr.update( visible = len(samples) == 0 ),
    sample_tree_row: gr.update( visible = len(samples) > 0 ),
    **settings_out_dict
  }