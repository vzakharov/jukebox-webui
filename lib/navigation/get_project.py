from lib.model.params import hps
import lib.navigation.utils
import lib.ui.components.first
import lib.ui.components.metas
import lib.ui.components.general
import lib.ui.components.misc
import lib.ui.components.navigation
import lib.ui.components.project
import lib.ui.components.upsampling
import lib.ui.components.main
from .get_samples import get_samples
from .utils import is_new
import lib.ui.components as UI
from .utils import loaded_settings
from params import base_path

import gradio as gr
import yaml

import os

def get_project(project_name, routed_sample_id):

  global base_path, loaded_settings

  is_this_new = is_new(project_name)

  # Start with default values for project settings
  settings_out_dict = {
    lib.ui.components.metas.artist: 'Unknown',
    lib.ui.components.metas.genre: 'Unknown',
    lib.ui.components.metas.lyrics: '',
    lib.ui.components.project.generation_length: 1,
    lib.ui.components.project.temperature: 0.98,
    lib.ui.components.project.n_samples: 2,
    lib.ui.components.navigation.sample_tree: None,
    lib.ui.components.upsampling.genre_for_upsampling_left_channel: 'Unknown',
    lib.ui.components.upsampling.genre_for_upsampling_center_channel: 'Unknown',
    lib.ui.components.upsampling.genre_for_upsampling_right_channel: 'Unknown',
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
          if key in lib.navigation.utils.inputs_by_name and lib.navigation.utils.inputs_by_name[key] in lib.ui.components.project.project_settings:

            input = lib.navigation.utils.inputs_by_name[key]

            # If the value is an integer (i) but the element is an instance of gr.components.Radio or gr.components.Dropdown, take the i-th item from the choices
            if isinstance(value, int) and isinstance(input, (gr.components.Radio, gr.components.Dropdown)):
              print(f'Converting {key} value {value} to {input.choices[value]}')
              value = input.choices[value]

            settings_out_dict[getattr(UI, key)] = value

          else:
            print(f'Warning: {key} is not a valid project setting')

    # Write the last project name to settings.yaml
    with open(f'{base_path}/settings.yaml', 'w') as f:
      print(f'Saving {project_name} as last project...')
      yaml.dump({'last_project': project_name}, f)
      print('Saved to settings.yaml')

    settings_out_dict[ lib.ui.components.misc.getting_started_column ] = gr.update(
      visible = False
    )

    samples = get_samples(project_name, settings_out_dict[ lib.ui.components.navigation.show_leafs_only ] if lib.ui.components.navigation.show_leafs_only in settings_out_dict else False)

    sample = routed_sample_id or (
      (
        settings_out_dict[ lib.ui.components.navigation.sample_tree ] or samples[0]
      ) if len(samples) > 0 else None
    )

    settings_out_dict[ lib.ui.components.navigation.sample_tree ] = gr.update(
      choices = samples,
      value = sample
    )

  return {
    lib.ui.components.general.create_project_box: gr.update( visible = is_this_new ),
    lib.ui.components.general.settings_box: gr.update( visible = not is_this_new ),
    lib.ui.components.main.workspace_column: gr.update( visible = not is_this_new  ),
    lib.ui.components.navigation.sample_box: gr.update( visible = sample is not None ),
    lib.ui.components.first.first_generation_row: gr.update( visible = len(samples) == 0 ),
    lib.ui.components.navigation.sample_tree_row: gr.update( visible = len(samples) > 0 ),
    **settings_out_dict
  }