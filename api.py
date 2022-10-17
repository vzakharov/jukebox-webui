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
  open('dist_installed', 'w').close()


# If the base folder doesn't exist, create it
if not os.path.isdir(base_path):
  os.makedirs(base_path)

with gr.Blocks() as ui:

  last_error = None

  def apified(fn):
    
    def apified_fn(*inputs):

      global last_error

      try:

        print(f'Calling {fn.__name__}...')

        result = fn(*inputs)

        print(f'API call successful: {result}')
        last_error = None
        
        return result

      except Exception as e:

        print(f'API call failed: {e}')

        last_error = str(e)

        raise e
      
    return apified_fn

  error_message = gr.Textbox(
    label = 'Error message (if any)'
  )

  def get_error_message():
    global last_error
    return last_error

  gr.Button('Get error message').click(
    inputs = None,
    outputs = error_message,
    fn = get_error_message,
    api_name = 'get-error-message'
  )

  string_list = gr.Dropdown(
    label = 'String list'
  )

  def get_projects():
  
    global base_path, string_list

    print(f'Getting project list for {base_path}...')

    project_names = []
    for folder in os.listdir(base_path):
      if os.path.isdir(base_path+'/'+folder) and not folder.startswith('_'):
        project_names.append(folder)
    # Sort project names alphabetically
    project_names.sort()
    # Add "CREATE NEW" option in the beginning
    return {
      string_list: gr.update(choices = project_names)
    }

    
  gr.Button('Get projects').click(
    inputs = None,
    outputs = string_list,
    fn = apified(get_projects),
    api_name = 'get-projects',
  )

  def create_project(name):

    global base_path

    path = f'{base_path}/{name}'
    assert not os.path.isdir(path), f'Project {name} already exists'

    os.makedirs(path)

    assert os.path.isdir(path), f'Project {name} could not be created'
  
  string_input = gr.Textbox(label = 'String input')

  gr.Button('Create project').click(
    inputs = string_input,
    outputs = None,
    fn = apified(create_project),
    api_name = 'create-project',
  )

ui.launch( share = share_gradio, debug = debug_gradio )  
