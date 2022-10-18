# Check if it's a Colab notebookby checking if google.colab package is available
try:
  import google.colab
  print('Running on Colab')
  is_colab = True
except:
  print('Not running on Colab')
  is_colab = False
  

colab_path = '/content/drive/My Drive/JukeboxGo' #@param{type:'string'}
local_path = 'G:/Мой диск/JukeboxGo'
base_path = colab_path if is_colab else local_path

share_gradio = True #@param{type:'boolean'}
debug_gradio = True #@param{type:'boolean'}

import random
import gradio as gr
import json
import os
import glob
import urllib.request
import re
import torch as t
import librosa

import yaml

# Dump strings as literal blocks in YAML
yaml.add_representer(str, lambda dumper, value: dumper.represent_scalar('tag:yaml.org,2002:str', value, style='|'))


if is_colab:

  import jukebox

  from jukebox.make_models import make_vqvae, make_prior, MODELS
  from jukebox.hparams import Hyperparams, setup_hparams
  from jukebox.utils.dist_utils import setup_dist_from_mpi
  from jukebox.sample import load_prompts, sample_partial_window

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

def get_projects():
  
  global base_path

  print(f'Getting project list for {base_path}...')

  project_names = []
  for folder in os.listdir(base_path):
    if os.path.isdir(base_path+'/'+folder) and not folder.startswith('_'):
      project_names.append(folder)
  # Sort project names alphabetically
  project_names.sort()

  print(f'Found {len(project_names)} projects: {project_names}')

  # Add "CREATE NEW" option in the beginning
  return ['CREATE NEW'] + project_names

def create_project(name):

  global base_path

  print(f'Creating project {name}...')

  os.makedirs(f'{base_path}/{name}')

  print(f'Project {name} created!')

  return get_projects()

calculated_metas = {}
loaded_settings = {}

class UI:

  ### General

  project_name = gr.Dropdown(
    label = 'Project name',
    choices = get_projects()
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

  general_inputs = [ project_name ]

  print('General inputs:', general_inputs)

  ### Project-specific

  ## Metas (artist, genre, lyrics)
  artist = gr.Dropdown(
    label = 'Artist'
  )

  genre = gr.Dropdown(
    label = 'Genre'
  )

  lyrics = gr.Textbox(
    label = 'Lyrics',
    max_lines = 5
  )

  total_duration = gr.Slider(
    label = 'Duration',
    minimum = 60,
    maximum = 600,
    step = 10
  )

  metas = [ artist, genre, lyrics, total_duration ]

  calculate_model_button = gr.Button(
    'Calculate model',
    variant = 'primary'
  )

  generation_box = gr.Box(
    visible = False
  )

  project_inputs = [ *metas ]

  print('Project inputs:', project_inputs)

  # Combine all IO components as a list in 'all'
  all_inputs = general_inputs + project_inputs

  # Create input_names where keys are the attributes and values are the names
  input_names = { input: name for name, input in locals().items() if isinstance(input, gr.components.FormComponent) }
  print('Input names:', input_names)

  inputs_by_name = { name: input for name, input in locals().items() if isinstance(input, gr.components.FormComponent) }
  print('Inputs by name:', inputs_by_name)


with gr.Blocks() as app:
  
  with gr.Row():

    with gr.Column( scale = 1 ):

      UI.project_name.render()

      def set_project(project_name):

        global base_path, loaded_settings

        is_new = project_name == 'CREATE NEW'

        # Start with default values for project settings
        settings_out_dict = {
          UI.artist: 'Unknown',
          UI.genre: 'Unknown',
          UI.lyrics: '',
          UI.total_duration: 200,
        }

        # If not new, load the settings from settings.yaml in the project folder, if it exists
        if not is_new:

          print(f'Loading settings for {project_name}...')

          settings_path = f'{base_path}/{project_name}/{project_name}.yaml'
          if os.path.isfile(settings_path):
            with open(settings_path, 'r') as f:
              loaded_settings = yaml.load(f, Loader=yaml.FullLoader)
              print(f'Loaded settings for {project_name}: {loaded_settings}')

              # Go through all the settings and set the value for settings_out_dict where the key is the element itself
              for key, value in loaded_settings.items():
                if key in UI.inputs_by_name and UI.inputs_by_name[key] in UI.project_inputs:
                  print(f'Found setting {key} with value {value}')
                  settings_out_dict[getattr(UI, key)] = value
                else:
                  print(f'Warning: {key} is not a valid project setting')
        
          print('Valid settings:', settings_out_dict)

          # Write the last project name to settings.yaml
          with open(f'{base_path}/settings.yaml', 'w') as f:
            print(f'Saving {project_name} as last project...')
            yaml.dump({'last_project': project_name}, f)
            print('Saved to settings.yaml')

        return {
          UI.create_project_box: gr.update( visible = is_new ),
          UI.project_box: gr.update( visible = not is_new and project_name != '' ),
          **settings_out_dict
        }
      
      UI.project_name.change(
        inputs = UI.project_name,
        outputs = [ UI.create_project_box, UI.project_box, *UI.project_inputs ],
        fn = set_project,
        api_name = 'set-project'
      )

      UI.create_project_box.render()

      with UI.create_project_box:

        UI.new_project_name.render()

        # When a project is created, create a subfolder for it and update the project list.

        create_args = {
          'inputs': UI.new_project_name,
          'outputs': UI.project_name,
          'fn': lambda name: gr.update( choices = create_project(name), value = name )
        }

        UI.new_project_name.submit( **create_args )
        gr.Button('Create project').click( **create_args )

      UI.project_box.render()

      with UI.project_box:

        def save_project_settings(project_name, *project_input_values):

          print(f'Saving settings for {project_name}...')
          print(f'Project input values: {project_input_values}')

          # Go through all UI attributes and add the ones that are in the project settings to a dictionary
          settings = {}

          for i in range(len(UI.project_inputs)):
            settings[UI.input_names[UI.project_inputs[i]]] = project_input_values[i]
          
          print(f'Settings: {settings}')

          # If the settings are different from the loaded settings, save them to the project folder

          if settings != loaded_settings:

            with open(f'{base_path}/{project_name}/{project_name}.yaml', 'w') as f:
              yaml.dump(settings, f)
              print(f'Saved settings to {base_path}/{project_name}/{project_name}.yaml')
          
          else:
            print('Settings are the same as loaded settings, not saving.')

        for component in UI.project_inputs:

          component.render()

          # Whenever a project setting is changed, save all the settings to settings.yaml in the project folder
          inputs = [ UI.project_name, *UI.project_inputs ]

          print(f'Inputs for {component}: {inputs}')

          # Use the "blur" method if available, otherwise use "change"
          handler_name = 'blur' if hasattr(component, 'blur') else 'change'
          handler = getattr(component, handler_name)
          print(f'Using {handler_name} handler for {component}')

          handler(
            inputs = inputs,
            outputs = None,
            fn = save_project_settings
          )

          def set_metas_changed(artist, genre, lyrics, total_duration):

            global calculated_metas

            metas_changed = calculated_metas != {
              'artist': artist,
              'genre': genre,
              'lyrics': lyrics,
              'total_duration': total_duration
            }

            return {
              UI.calculate_model_button: gr.update( visible = metas_changed ),
              UI.generation_box: gr.update( visible = not metas_changed )
            }

          # When a meta setting is changed, show the calculate model button and hide the generation box
          # Use blur/change depending on the component
          if component in UI.metas:
            handler(
              inputs = UI.metas,
              outputs = [ UI.calculate_model_button, UI.generation_box ],
              fn = set_metas_changed,
            )

        UI.calculate_model_button.render()

        # When the calculate model button is clicked, calculate the model and set the metas_changed flag to False
        def calculate_model(artist, genre, lyrics, total_duration):

          global calculated_metas

          # Calculate the model
          # (to be implemented)

          calculated_metas = {
            'artist': artist,
            'genre': genre,
            'lyrics': lyrics,
            'total_duration': total_duration
          }

          return {
            UI.calculate_model_button: gr.update( visible = False, value = 'Recalculate model' ),
            UI.generation_box: gr.update( visible = True )
          }
        
        UI.calculate_model_button.click(
          inputs = UI.metas,
          outputs = [ UI.calculate_model_button, UI.generation_box ],
          fn = calculate_model
        )
    
    with gr.Column( scale = 3 ):

      UI.generation_box.render()

      with UI.generation_box:

        gr.Markdown('Generation inputs will go here')
        # (to be implemented)

  # If the app is loaded and the list of projects is empty, set the project list to CREATE NEW. Otherwise, load the last project from settings.yaml, if it exists.
  def get_last_project():

    print('Getting last project...')

    if len(UI.project_name.choices) == 1:
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
          return ''

  app.load(
    get_last_project,
    inputs = None,
    outputs = UI.project_name
  )

  app.launch( share = share_gradio, debug = debug_gradio )