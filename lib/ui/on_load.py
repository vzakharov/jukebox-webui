import os
import re

import gradio as gr
import yaml

from lib.lists import get_list
from lib.navigation.get_projects import get_projects
from params import base_path
from UI.general import project_name
from UI.main import main_window
from UI.metas import artist, genre_dropdown
from UI.misc import (getting_started_column, separate_tab_link,
                     separate_tab_warning)
from UI.navigation import routed_sample_id
from UI.upsampling import (genre_center_channel, genre_left_channel,
                           genre_right_channel)


def on_load( href, query_string, error_message ):

  if error_message:
    print(f'Please open this app in a separate browser tab: {href}')
    print(f'Error message from the client (for debugging only; you can ignore this): {error_message}')

    return {
      separate_tab_warning: gr.update(
        visible = True,
      ),
      separate_tab_link: href,
      main_window: gr.update(
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
    project_name: gr.update(
      choices = projects,
      value = project_name,
    ),
    routed_sample_id: sample_id,
    artist: gr.update(
      choices = get_list('artist'),
    ),
    genre_dropdown: gr.update(
      choices = get_list('genre'),
    ),
    getting_started_column: gr.update(
      visible = len(projects) == 1
    ),
    separate_tab_warning: gr.update(
      visible = False
    ),
    main_window: gr.update(
      visible = True
    ),
    genre_left_channel: gr.update(
      choices = get_list('genre')
    ),
    genre_center_channel: gr.update(
      choices = get_list('genre'),
    ),
    genre_right_channel: gr.update(
      choices = get_list('genre'),
    ),
  }