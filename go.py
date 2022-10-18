base_path = '/content/drive/My Drive/JukeboxGo'  #@param{type:'string'}
share_gradio = True #@param{type:'boolean'}
debug_gradio = True #@param{type:'boolean'}

import random
import gradio as gr
import json
import os
import glob
import urllib.request
import yaml
import re
import torch as t
import librosa

import jukebox

from jukebox.make_models import make_vqvae, make_prior, MODELS
from jukebox.hparams import Hyperparams, setup_hparams
from jukebox.utils.dist_utils import setup_dist_from_mpi
from jukebox.sample import load_prompts, sample_partial_window

# Dump strings as literal blocks in YAML
yaml.add_representer(str, lambda dumper, value: dumper.represent_scalar('tag:yaml.org,2002:str', value, style='|'))

### Model

hps = Hyperparams()
hps.sr = 44100
hps.levels = 3
hps.hop_fraction = [ 0.5, 0.5, 0.125 ]

raw_to_tokens = 128
chunk_size = 16
model = '5b_lyrics'
lower_batch_size = 16
lower_level_chunk_size = 32

vqvae = None
priors = None
top_prior = None

# If there is no file named 'dist_installed' in the root folder, install the package
if not os.path.isfile('dist_installed'):
  rank, local_rank, device = setup_dist_from_mpi()

  # Create a 'dist_installed' file to avoid installing the package again when the app is restarted
  open('dist_installed', 'a').close()


# If the base folder doesn't exist, create it
if not os.path.isdir(base_path):
  os.makedirs(base_path)

### Projects

def get_project_list():
  
  global base_path

  print(f'Getting project list for {base_path}...')

  project_names = []
  for folder in os.listdir(base_path):
    if os.path.isdir(base_path+'/'+folder) and not folder.startswith('_'):
      project_names.append(folder)
  # Sort project names alphabetically
  project_names.sort()
  # Add "CREATE NEW" option in the beginning
  return ['CREATE NEW'] + project_names

def create_project(name):

  global base_path

  print(f'Creating project {name}...')

  os.makedirs(f'{base_path}/{name}')

  return get_project_list()


class UI:

  projects_list = gr.Dropdown(
    label = 'Pick a project',
    choices = get_project_list()
  )

  create_project_box = gr.Box(
    visible = False
  )

  new_project_name = gr.Textbox(
    label = 'New project name'
  )

  project_box = gr.Box(
    visible = False
  )

with gr.Blocks() as app:

  UI.projects_list.render()

  # If "CREATE NEW" is selected, show the create_project_box. Otherwise, show the project_box  
  # def create_or_open_project(name):

  #   print(f'Toggling boxes for {name}...')

  #   return {
  #     UI.create_project_box: gr.update( visible = name == 'CREATE NEW' ),
  #     UI.project_box: gr.update( visible = name != 'CREATE NEW' and name != '' )
  #   }
  
  UI.projects_list.change(
    inputs = UI.projects_list,
    outputs = [ UI.create_project_box, UI.project_box ],
    fn = lambda name: {
      UI.create_project_box: gr.update( visible = name == 'CREATE NEW' ),
      UI.project_box: gr.update( visible = name != 'CREATE NEW' and name != '' )
    }
  )

  UI.create_project_box.render()

  with UI.create_project_box:

    UI.new_project_name.render()

    # When a project is created, create a subfolder for it and update the project list.

    create_args = {
      'inputs': UI.new_project_name,
      'outputs': UI.projects_list,
      'fn': lambda name: gr.update( choices = create_project(name), value = name )
    }

    UI.new_project_name.submit( **create_args )
    gr.Button('Create project').click( **create_args )

  UI.project_box.render()

  with UI.project_box:

    gr.Markdown('## Project')

  # If app is loaded and the list of projects is empty, set the project list to CREATE NEW. Otherwise, load the last project from settings.json, if it exists.
  def get_last_project():

    if len(UI.projects_list.choices) == 1:
      return 'CREATE NEW'

    elif os.path.isfile(f'{base_path}/settings.json'):
      with open(f'{base_path}/settings.json', 'r') as f:
        settings = json.load(f)
        return settings['last_project']

  app.load(
    inputs = [],
    outputs = UI.projects_list,
    fn = get_last_project,
    api_name = 'get-last-project'
  )

  app.launch( share = share_gradio, debug = debug_gradio )