from params import base_path

import yaml

import os

def get_last_project(projects):
  if len(projects) == 1:
    return 'CREATE NEW'

  elif os.path.isfile(f'{base_path}/settings.yaml'):
    with open(f'{base_path}/settings.yaml', 'r') as f:
      settings = yaml.load(f, Loader=yaml.FullLoader)
      print(f'Loaded settings: {settings}')
      if 'last_project' in settings:
        print(f'Last project: {settings["last_project"]}')
        return settings['last_project']
      else:
        print('No last project found.')
        return projects[0]