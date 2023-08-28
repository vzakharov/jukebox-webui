from lib.utils import convert_name
from lib.navigation.get_projects import get_projects
from params import base_path

import gradio as gr

import os

def create_project(name):

  global base_path

  name = convert_name(name)

  print(f'Creating project {name}...')

  os.makedirs(f'{base_path}/{name}')

  print(f'Project {name} created!')

  return gr.update(
    choices = get_projects(),
    value = name
  )