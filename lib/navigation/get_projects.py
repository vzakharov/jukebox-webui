from params import base_path


import os


def get_projects(include_new = True):

  global base_path

  # print(f'Getting project list for {base_path}...')

  project_names = []
  for folder in os.listdir(base_path):
    if os.path.isdir(base_path+'/'+folder) and not folder.startswith('_'):
      project_names.append(folder)
  # Sort project names alphabetically
  project_names.sort()

  print(f'Found {len(project_names)} projects: {project_names}')

  if include_new:
    project_names = ['CREATE NEW', *project_names]

  return project_names