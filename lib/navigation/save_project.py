from lib.navigation.utils import is_new
from lib.ui.UI import UI
from lib.navigation.utils import loaded_settings
from params import base_path

import yaml

def save_project(project_name, *project_input_values):

  if is_new(project_name):
    return

  # print(f'Saving settings for {project_name}...')
  # print(f'Project input values: {project_input_values}')

  # Go through all UI attributes and add the ones that are in the project settings to a dictionary
  settings = {}

  for i in range(len(UI.project_settings)):
    settings[UI.input_names[UI.project_settings[i]]] = project_input_values[i]

  # print(f'Settings: {settings}')

  # If the settings are different from the loaded settings, save them to the project folder

  if settings != loaded_settings:

    with open(f'{base_path}/{project_name}/{project_name}.yaml', 'w') as f:
      yaml.dump(settings, f)
      print(f'Saved settings to {base_path}/{project_name}/{project_name}.yaml: {settings}')