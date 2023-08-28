from params import base_path


import yaml


import os

custom_parents = None

def get_custom_parents(project_name, force_reload=False):

  global base_path, custom_parents

  if not custom_parents or custom_parents['project_name'] != project_name or force_reload:
    print('Loading custom parents...')
    custom_parents = {}
    filename = f'{base_path}/{project_name}/{project_name}-parents.yaml'
    if os.path.exists(filename):
      print(f'Found {filename}')
      with open(filename) as f:
        loaded_dict = yaml.load(f, Loader=yaml.FullLoader)
        print(f'Loaded as {loaded_dict}')
        # Add project_name to the beginning of every key and value in the dictionary
        custom_parents = { f'{project_name}-{k}': f'{project_name}-{v}' for k, v in loaded_dict.items() }

    custom_parents['project_name'] = project_name

    print(f'Custom parents: {custom_parents}')

  return custom_parents