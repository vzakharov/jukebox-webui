# Check if it's a Colab notebookby checking if google.colab package is available
try:
  import google.colab
  print('Running on Colab')
  is_colab = True
except:
  print('Not running on Colab')
  is_colab = False
  

colab_path = '/content/drive/My Drive/jukebox-webui' #@param{type:'string'}
local_path = 'G:/Мой диск/jukebox-webui'
colab_data_path = '/content/drive/My Drive/jukebox-webui/_data' #@param{type:'string'}
local_data_path = 'G:/Мой диск/jukebox-webui/_data'

base_path = colab_path if is_colab else local_path
data_path = colab_data_path if is_colab else local_data_path

share_gradio = True #@param{type:'boolean'}
debug_gradio = True #@param{type:'boolean'}

import random
from time import sleep
import gradio as gr
import json
import os
import shutil
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
  import jukebox.utils.dist_adapter as dist

  from jukebox.make_models import make_vqvae, make_prior, MODELS
  from jukebox.hparams import Hyperparams, setup_hparams, REMOTE_PREFIX
  from jukebox.utils.dist_utils import setup_dist_from_mpi
  from jukebox.utils.remote_utils import download
  from jukebox.utils.torch_utils import empty_cache
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

reload_all = False #@param{type:'boolean'}
reload_dist = False #@param{type:'boolean'}
try:
  assert not reload_dist and not reload_all
  rank, local_rank, device
  print('Dist already setup')
except:
  rank, local_rank, device = setup_dist_from_mpi()
  print(f'Dist setup: rank={rank}, local_rank={local_rank}, device={device}')

# Monkey patch jukebox.make_models.load_checkpoint to load cached checkpoints from local_data_path instead of '~/.cache'
reload_monkey_patch = False #@param{type:'boolean'}
try:
  assert not reload_monkey_patch and not reload_all
  monkey_patched_load_checkpoint
  print('load_checkpoint already monkey patched')
except:

  def download_to_cache(remote_path, local_path):
    print(f'Caching {remote_path} to {local_path}')
    if not os.path.exists(os.path.dirname(local_path)):
      print(f'Creating directory {os.path.dirname(local_path)}')
      os.makedirs(os.path.dirname(local_path))
    if not os.path.exists(local_path):
      print('Downloading...')
      download(remote_path, local_path)
      print('Done.')
    else:
      print('Already cached.')

  def monkey_patched_load_checkpoint(path):
    global data_path
    restore = path
    if restore.startswith(REMOTE_PREFIX):
        remote_path = restore
        local_path = os.path.join(data_path, remote_path[len(REMOTE_PREFIX):])
        if dist.get_rank() % 8 == 0:
            download_to_cache(remote_path, local_path)
        restore = local_path
    dist.barrier()
    checkpoint = t.load(restore, map_location=t.device('cpu'))
    print("Restored from {}".format(restore))
    return checkpoint

  # Download jukebox/models/5b/vqvae.pth.tar and jukebox/models/5b_lyrics/prior_level_2.pth.tar right away to avoid downloading them on the first run
  for model in ['jukebox/models/5b/vqvae.pth.tar', 'jukebox/models/5b_lyrics/prior_level_2.pth.tar']:
    download_to_cache(f'{REMOTE_PREFIX}{model}', os.path.join(data_path, model))

  jukebox.make_models.load_checkpoint = monkey_patched_load_checkpoint

  print('Monkey patched load_checkpoint and downloaded the checkpoints')


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

def get_meta(what):
  items = []
  print(f'Getting {what} list...')
  with urllib.request.urlopen(f'https://raw.githubusercontent.com/openai/jukebox/master/jukebox/data/ids/v2_{what}_ids.txt') as f:
    for line in f:
      item = line.decode('utf-8').split(';')[0]
      item = item.replace('_', ' ').title()
      items.append(item)
  items.sort()
  print(f'Loaded {len(items)} {what}s.')
  return items

try:
  calculated_metas
  print('Using existing calculated_metas')
except:
  calculated_metas = {}

loaded_settings = {}

class UI:

  ### General

  project_name = gr.Dropdown(
    label = 'Project',
    choices = get_projects()
  )

  create_project_box = gr.Box(
    visible = False
  )

  new_project_name = gr.Textbox(
    label = 'Project name',
    placeholder = 'lowercase-digits-and-dashes-only'
  )

  project_box = gr.Box(
    visible = False
  )

  general_inputs = [ project_name ]

  print('General inputs:', general_inputs)

  ### Project-specific

  ## Metas (artist, genre, lyrics)
  artist = gr.Dropdown(
    label = 'Artist',
    choices = get_meta('artist')
  )

  genre = gr.Dropdown(
    label = 'Genre',
    choices = get_meta('genre')
  )

  lyrics = gr.Textbox(
    label = 'Lyrics',
    max_lines = 5,
    placeholder = 'Shift+Enter for new line'
  )

  total_duration = gr.Slider(
    label = 'Duration, sec',
    minimum = 60,
    maximum = 600,
    step = 10
  )

  metas = [ artist, genre, lyrics, total_duration ]

  calculate_model_button = gr.Button(
    'Calculate model',
    variant = 'primary'
  )

  generation_text = gr.Markdown(
    'Calculate the model to start creating.'
  )

  generation_box = gr.Box(
    visible = False
  )

  generation_length = gr.Slider(
    label = 'Generation length, sec',
    minimum = 0.5,
    maximum = 10,
    step = 0.25
  )

  generated_audio = gr.Audio(
    label = 'Generated audio',
    visible = False
  )

  project_inputs = [ *metas, generation_length ]

  print('Project inputs:', project_inputs)

  # status_bar = gr.Textbox(
  #   label = 'Status',
  # )

  # Combine all IO components as a list in 'all'
  all_inputs = general_inputs + project_inputs

  input_names = { input: name for name, input in locals().items() if isinstance(input, gr.components.FormComponent) }

  inputs_by_name = { name: input for name, input in locals().items() if isinstance(input, gr.components.FormComponent) }


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

        # Wehn the new project name is unfocused, convert it to lowercase and replace non-alphanumeric characters with dashes
        def convert_name(name):
          return re.sub(r'[^a-z0-9]+', '-', name.lower())

        UI.new_project_name.blur(
          inputs = UI.new_project_name,
          outputs = UI.new_project_name,
          fn = convert_name,
        )

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

        # When a project is created, create a subfolder for it and update the project list.
        create_args = dict(
          inputs = UI.new_project_name,
          outputs = UI.project_name,
          fn = create_project,
        )

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

        for component in UI.metas:
          component.render()

        for component in UI.project_inputs:

          # Whenever a project setting is changed, save all the settings to settings.yaml in the project folder
          inputs = [ UI.project_name, *UI.project_inputs ]

          # Use the "blur" method if available, otherwise use "change"
          handler_name = 'blur' if hasattr(component, 'blur') else 'change'
          handler = getattr(component, handler_name)

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
              UI.generation_text: gr.update( visible = metas_changed ),
              UI.generation_box: gr.update( visible = not metas_changed )
            }

          # When a meta setting is changed, show the calculate model button and hide the generation box
          # Use blur/change depending on the component
          if component in UI.metas:
            handler(
              inputs = UI.metas,
              outputs = [ UI.calculate_model_button, UI.generation_text, UI.generation_box ],
              fn = set_metas_changed,
            )

        UI.calculate_model_button.render()

        # When the calculate model button is clicked, calculate the model and set the metas_changed flag to False
        def calculate_model(artist, genre, lyrics, total_duration):

          global calculated_metas
          global hps, raw_to_tokens, chunk_size
          global vqvae, priors, top_prior, device
          global metas, labels

          n_samples = 1

          try:
            calculated_duration = calculated_metas['total_duration']
          except:
            calculated_duration = None

          hps.sample_length = int(total_duration * hps.sr // raw_to_tokens) * raw_to_tokens

          if total_duration != calculated_duration:

            print(f'Duration {total_duration} is not equal to duration used by model {calculated_duration}, recalculating model...')

            try:
              print('Deleting vqvae & top_prior...')
              del vqvae
              del top_prior
              empty_cache()
              print('Deleted.')
            except:
              print('Not found; skipping.')

            vqvae, *priors = MODELS[model]
            
            vqvae = make_vqvae(setup_hparams(vqvae, dict(sample_length = hps.sample_length)), device)

            top_prior = make_prior(setup_hparams(priors[-1], dict()), vqvae, device)
          
          metas = [dict(
            artist = artist,
            genre = genre,
            total_length = hps.sample_length,
            offset = 0,
            lyrics = lyrics,
          )] * n_samples

          labels = top_prior.labeller.get_batch_labels(metas, device)

          calculated_metas = {
            'artist': artist,
            'genre': genre,
            'lyrics': lyrics,
            'total_duration': total_duration
          }

          return {
            UI.calculate_model_button: gr.update( visible = False, value = 'Recalculate model' ),
            UI.generation_text: gr.update( 
              visible = False,
              value = 'You need to recalculate the model if you change the artist, genre, lyrics or total duration.'
            ),
            UI.generation_box: gr.update( visible = True )
          }
        
        UI.calculate_model_button.click(
          inputs = UI.metas,
          outputs = [ UI.calculate_model_button, UI.generation_text, UI.generation_box ],
          fn = calculate_model,
          api_name = 'calculate-model',
        )
    
    with gr.Column( scale = 3 ):

      UI.generation_text.render()
      UI.generation_box.render()

      with UI.generation_box:

        UI.generation_length.render()

        generate_button = gr.Button(
          'Generate',
          variant = 'primary',
        )

        generating_spinner = gr.Markdown('')
        UI.generated_audio.render()

        def seconds_to_tokens(sec):

          global hps, raw_to_tokens, chunk_size

          tokens = sec * hps.sr // raw_to_tokens
          tokens = ( (tokens // chunk_size) + 1 ) * chunk_size
          return int(tokens)

        def write_files(base_path, project_name, zs, wav, parent_sample_id=''):

          # 1. Scan project folder for [project_name][-parent_sample_id]-[comma-separated child ids].zs
          # 2. Take the highest child id and add 1 to it (call it first_new_child_id)
          # 3. Create [project_name][-parent_sample_id]-[first_new_child_id,first_new_child_id+1,first_new_child_id+2,etc.].zs depending on how many samples we have (based on the shape of zs)

          global hps
          n_samples = wav.shape[0]

          base_filename = f'{base_path}/{project_name}/{project_name}'
          if parent_sample_id:
            base_filename += f'-{parent_sample_id}'

          files = glob.glob(f'{base_filename}-*.zs')
          if not files:
            first_new_child_id = 1
          else:
            existing_ids = []
            for f in files:
              # Extract the child ids from the filename by splitting the part after the last dash by the comma
              child_ids = f.split('-')[-1].split('.')[0].split(',')
              child_ids = [ int(c) for c in child_ids ]
              existing_ids += child_ids
            first_new_child_id = max(existing_ids) + 1

            zs_filename = f"{base_filename}-{','.join([ str(i+first_new_child_id) for i in range(n_samples) ])}"
            t.save(zs, zs_filename)

            for i in range(n_samples):
              wav_filename = f"{base_filename}-{first_new_child_id+i}.wav"
              librosa.output.write_wav(wav_filename, wav[i], hps.sr)
            
            # Return all the new filenames
            return [ f'{zs_filename}.zs', *[ f'{base_filename}-{first_new_child_id+i}.wav' for i in range(n_samples) ] ]

        def generate(project_name, generation_length):

          print(f'Generating {generation_length} seconds for {project_name}...')

          n_samples = 1
          temperature = 0.98
          # The above is to be moved to parameters/the UI

          global base_path
          global hps, raw_to_tokens, chunk_size, lower_batch_size, lower_level_chunk_size
          global top_prior

          zs = [ t.zeros(n_samples, 0, dtype=t.long, device='cuda') for _ in range(3) ]
          
          tokens_to_sample = seconds_to_tokens(generation_length)
          sampling_kwargs = dict(
            temp=temperature, fp16=True, max_batch_size=lower_batch_size,
            chunk_size=lower_level_chunk_size
          )

          zs = sample_partial_window(zs, labels, sampling_kwargs, 2, top_prior, tokens_to_sample, hps)
          print('- zs generated.')

          wav = vqvae.decode(zs[2:], start_level=2).cpu().numpy()
          print('- wav generated.')

          filenames = write_files(base_path, project_name, zs, wav)
          print(f'- Files written: {filenames}')

          return {
            generating_spinner: gr.update(),
            UI.generated_audio: gr.update(
              visible = True,
              value = filenames[1],
            ),
          }

        generate_button.click(
          inputs = [ UI.project_name, UI.generation_length ],
          outputs = [ generating_spinner, UI.generated_audio ],
          fn = generate,
          api_name = 'generate',
        )

  # with gr.Row():

  #   UI.status_bar.render()


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