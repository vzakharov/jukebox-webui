from lib.lists import get_list
from .UI import UI
from lib.navigation.get_projects import get_projects
from params import base_path

import gradio as gr
import yaml

import os
import re

def on_load( href, query_string, error_message ):

  if error_message:
    print(f'Please open this app in a separate browser tab: {href}')
    print(f'Error message from the client (for debugging only; you can ignore this): {error_message}')

    return {
      UI.separate_tab_warning: gr.update(
        visible = True,
      ),
      UI.separate_tab_link: href,
      UI.main_window: gr.update(
        visible = False,
      ),
    }

  projects = get_projects()

  def get_last_project():
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

  # If there is a query string, it will be of the form project_name-sample_id or project_name
  if query_string:
    print(f'Query string: {query_string}')
    if '-' in query_string:
      project_name, sample_id = re.match('^(.*?)-(.*)$', query_string).groups()
      sample_id = f'{project_name}-{sample_id}'
      print(f'Routed to project {project_name} and sample {sample_id}')
    else:
      project_name = query_string
      sample_id = None
      print(f'Routed to project {project_name}')
  else:
    project_name = get_last_project()
    sample_id = None

  return {
    UI.project_name: gr.update(
      choices = projects,
      value = project_name,
    ),
    UI.routed_sample_id: sample_id,
    UI.artist: gr.update(
      choices = get_list('artist'),
    ),
    UI.genre_dropdown: gr.update(
      choices = get_list('genre'),
    ),
    UI.getting_started_column: gr.update(
      visible = len(projects) == 1
    ),
    UI.separate_tab_warning: gr.update(
      visible = False
    ),
    UI.main_window: gr.update(
      visible = True
    ),
    UI.genre_for_upsampling_left_channel: gr.update(
      choices = get_list('genre')
    ),
    UI.genre_for_upsampling_center_channel: gr.update(
      choices = get_list('genre'),
    ),
    UI.genre_for_upsampling_right_channel: gr.update(
      choices = get_list('genre'),
    ),
  }