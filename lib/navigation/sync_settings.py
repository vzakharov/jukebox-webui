import os

import gradio as gr
import yaml

from lib.model.params import hps
from lib.ui.elements.misc import getting_started_column
from lib.ui.elements.navigation import sample_tree, show_leafs_only
from lib.ui.elements.project import project_settings
from params import base_path

from .get_samples import get_samples
from .utils import inputs_by_name

def sync_settings(project_name, routed_sample_id, settings_out_dict):
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

    return samples,sample