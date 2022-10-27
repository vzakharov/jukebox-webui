import os

try:

  is_colab # If is_colab is defined, we don't need to check again
  print('Re-running the cell')

except:
  
  try:

    from google.colab import drive
    drive.mount('/content/drive')

    !nvidia-smi
    !pip install git+https://github.com/openai/jukebox.git
    !pip install gradio
    # os.system('pip install git+https://github.com/openai/jukebox.git')
    # os.system('pip install gradio')
    # TODO: Fing a way to do the !stuff so that it runs in non-colab environments

    is_colab = True

  except:
    print('Not running on Colab')
    is_colab = False
  

total_duration = 200 #@param {type:"slider", min:60, max:300, step:10}

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
import shutil
import glob
import numpy as np
import urllib.request
import re
import torch as t
import librosa

import matplotlib
import matplotlib.pyplot as plt

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

raw_to_tokens = 128
chunk_size = 16
model = '5b_lyrics'
lower_batch_size = 16
lower_level_chunk_size = 32

hps = Hyperparams()
hps.sr = 44100
hps.levels = 3
hps.hop_fraction = [ 0.5, 0.5, 0.125 ]
hps.sample_length = int(total_duration * hps.sr // raw_to_tokens) * raw_to_tokens

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
  print('Checking monkey patch...')
  assert not reload_monkey_patch and not reload_all
  monkey_patched_load_checkpoint
  print('load_checkpoint already monkey patched.')
except:

  print('load_checkpoint not monkey patched; monkey patching...')

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
  for model_path in ['jukebox/models/5b/vqvae.pth.tar', 'jukebox/models/5b_lyrics/prior_level_2.pth.tar']:
    download_to_cache(f'{REMOTE_PREFIX}{model_path}', os.path.join(data_path, model_path))

  jukebox.make_models.load_checkpoint = monkey_patched_load_checkpoint

  print('Monkey patched load_checkpoint and downloaded the checkpoints')

try:
  vqvae, priors, top_prior
  print('Model already loaded.')
except:
  vqvae = None
  priors = None
  top_prior = None
  reload_prior = True
  print('Model not loaded; loading...')

reload_prior = False #@param{type:'boolean'}

try:
  calculated_duration
except:
  calculated_duration = 0

if total_duration != calculated_duration or reload_prior or reload_all:

  print(f'Loading vqvae and top_prior for duration {total_duration}...')

  try:
    del vqvae
    print('Deleted vqvae')
    del top_prior
    print('Deleted top_prior')
    empty_cache()
    print('Emptied cache')
  except:
    print('Either vqvae or top_prior is not defined; skipping deletion')

  vqvae, *priors = MODELS[model]

  vqvae = make_vqvae(setup_hparams(vqvae, dict(sample_length = hps.sample_length)), device)

  top_prior = make_prior(setup_hparams(priors[-1], dict()), vqvae, device)

  calculated_duration = total_duration


# If the base folder doesn't exist, create it
if not os.path.isdir(base_path):
  os.makedirs(base_path)

### Projects

def get_projects():
  
  global base_path

  # print(f'Getting project list for {base_path}...')

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
  # print(f'Getting {what} list...')
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

  project_box = gr.Accordion(
    label = "Settings",
    visible = False
  )

  general_settings = [ project_name ]

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

  metas = [ artist, genre, lyrics ]

  n_samples = gr.Slider(
    label = 'Number of samples',
    minimum = 1,
    maximum = 10,
    step = 1
  )

  temperature = gr.Slider(
    label = 'Temperature',
    minimum = 0,
    maximum = 1.5,
    step = 0.005
  )

  generation_length = gr.Slider(
    label = 'Generation length, sec',
    minimum = 0.5,
    maximum = 10,
    step = 0.1
  )

  generation_params = [ artist, genre, lyrics, n_samples, temperature, generation_length ]

  sample_tree = gr.Dropdown(
    label = 'Sample tree',
  )

  generate_first_button = gr.Button(
    'Generate',
    variant = 'primary'
  )

  picked_sample = gr.Radio(
    label = 'Sibling samples',
  )

  sample_box = gr.Box(
    visible = False
  )

  generated_audio = gr.Audio(
    label = 'Generated audio'
  )

  audio_waveform = gr.Plot(
    label = 'Waveform'
  )

  go_to_parent_button = gr.Button(
    value = '< parent',
  )

  go_to_children_button = gr.Button(
    value = '> children',
  )

  preview_just_the_last_n_sec = gr.Slider(
    label = 'Preview just the last ... seconds (0 to disable)',
    minimum = 0,
    maximum = 200,
    step = 0.1,
    value = 0
  )

  trim_to_n_sec = gr.Slider(
    label = 'Trim to ... seconds (0 to disable)',
    minimum = 0,
    maximum = 200,
    step = 0.05,
    value = 0
  )

  trim_button = gr.Button( 'Trim', visible = False )

  project_settings = [ *generation_params, sample_tree, generation_length, preview_just_the_last_n_sec ]

  input_names = { input: name for name, input in locals().items() if isinstance(input, gr.components.FormComponent) }

  inputs_by_name = { name: input for name, input in locals().items() if isinstance(input, gr.components.FormComponent) }


with gr.Blocks(
  css = """
    .gr-button {
      /* add margin to the button */
      margin: 5px 5px 5px 5px;
    }
  """,
  title = 'Jukebox Web UI',
) as app:
  
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
          UI.sample_tree: 'NONE',
          UI.generation_length: 1,
          UI.temperature: 0.98,
          UI.n_samples: 2
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
                if key in UI.inputs_by_name and UI.inputs_by_name[key] in UI.project_settings:
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
      
      def get_project_samples(project_name):

        choices = []
        for filename in os.listdir(f'{base_path}/{project_name}'):
          if re.match(r'.*\.zs?$', filename):
            id = filename.split('.')[0]
            choices += [ id ]
        
        # Sort by id, in descending order
        choices.sort(reverse = True)
        
        return [ 'NONE' ] + choices
      
      UI.project_name.change(
        inputs = UI.project_name,
        outputs = [ UI.create_project_box, UI.project_box, *UI.project_settings ],
        fn = set_project,
        api_name = 'set-project'
      )

      UI.project_name.change(
        inputs = UI.project_name,
        outputs = UI.sample_tree,
        fn = lambda project_name: {
          UI.sample_tree: gr.update( choices = get_project_samples(project_name) )
        },
        api_name = 'get-project-samples'
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

          for i in range(len(UI.project_settings)):
            settings[UI.input_names[UI.project_settings[i]]] = project_input_values[i]
          
          print(f'Settings: {settings}')

          # If the settings are different from the loaded settings, save them to the project folder

          if settings != loaded_settings:

            with open(f'{base_path}/{project_name}/{project_name}.yaml', 'w') as f:
              yaml.dump(settings, f)
              print(f'Saved settings to {base_path}/{project_name}/{project_name}.yaml')
          
          else:
            print('Settings are the same as loaded settings, not saving.')

        for component in UI.generation_params:
          component.render()

        for component in UI.project_settings:

          # Whenever a project setting is changed, save all the settings to settings.yaml in the project folder
          inputs = [ UI.project_name, *UI.project_settings ]

          # Use the "blur" method if available, otherwise use "change"
          handler_name = 'blur' if hasattr(component, 'blur') else 'change'
          handler = getattr(component, handler_name)

          handler(
            inputs = inputs,
            outputs = None,
            fn = save_project_settings
          )

    with gr.Column( scale = 3 ):

      UI.sample_tree.render()       
      UI.generate_first_button.render()
      UI.picked_sample.render()

      def seconds_to_tokens(sec):

        global hps, raw_to_tokens, chunk_size

        tokens = sec * hps.sr // raw_to_tokens
        tokens = ( (tokens // chunk_size) + 1 ) * chunk_size
        return int(tokens)

      def is_none_ish(string):
        return not string or string == 'NONE'

      def get_prefix(project_name, parent_sample_id):
        return f'{project_name if is_none_ish(parent_sample_id) else parent_sample_id}-'

      def get_children(project_name, parent_sample_id):

        global base_path

        prefix = get_prefix(project_name, parent_sample_id)
        child_ids = []
        for filename in os.listdir(f'{base_path}/{project_name}'):
          match = re.match(f'{prefix}(\d+)\\.zs?', filename)
          if match:
            child_ids += [ filename.split('.')[0] ]

        print(f'Children of {parent_sample_id}: {child_ids}')

        return child_ids
      
      def get_parent(project_name, sample_id):
        # Remove the project name and first dash from the sample id
        path = sample_id[ len(project_name) + 1: ].split('-')
        parent_sample_id = '-'.join([ project_name, *path[:-1] ]) if len(path) > 1 else None
        print(f'Parent of {sample_id}: {parent_sample_id}')
        return parent_sample_id
      
      def get_siblings(project_name, sample_id):

        return get_children(project_name, get_parent(project_name, sample_id))


      def generate(project_name, parent_sample_id, artist, genre, lyrics, n_samples, temperature, generation_length):

        print('Generating...')

        global total_duration
        global calculated_metas
        global hps, raw_to_tokens, chunk_size, lower_batch_size, lower_level_chunk_size
        global top_prior, device
        global metas, labels

        hps.n_samples = n_samples

        # If metas have changed, recalculate the metas
        if calculated_metas != dict( artist = artist, genre = genre, lyrics = lyrics ):

          print(f'Metas have changed, recalculating the model for {artist}, {genre}, {lyrics}')

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
            'lyrics': lyrics
          }

          print('Done recalculating the model')

        print(f'Generating {generation_length} seconds for {project_name}...')

        if is_none_ish(parent_sample_id):
          zs = [ t.zeros(n_samples, 0, dtype=t.long, device='cuda') for _ in range(3) ]
          print('No parent sample, generating from scratch')
        else:
          zs = t.load(f'{base_path}/{project_name}/{parent_sample_id}.z')
          print(f'Loaded parent sample {parent_sample_id} of shape {[ z.shape for z in zs ]}')
          # zs is a list of tensors of torch.Size([loaded_n_samples, n_tokens])
          # We need to turn it into a list of tensors of torch.Size([n_samples, n_tokens])
          # We do this by repeating the first sample of each tensor n_samples times
          zs = [ z[0].repeat(n_samples, 1) for z in zs ]
          print(f'Converted to shape {[ z.shape for z in zs ]}')
        
        tokens_to_sample = seconds_to_tokens(generation_length)
        sampling_kwargs = dict(
          temp=temperature, fp16=True, max_batch_size=lower_batch_size,
          chunk_size=lower_level_chunk_size
        )

        print(f'zs: {[ z.shape for z in zs ]}')
        zs = sample_partial_window(zs, labels, sampling_kwargs, 2, top_prior, tokens_to_sample, hps)
        print(f'Generated zs of shape {[ z.shape for z in zs ]}')

        wavs = vqvae.decode(zs[2:], start_level=2).cpu().numpy()
        print(f'Generated wavs of shape {wavs.shape}')

        # filenames = write_files(base_path, project_name, zs, wav)
        # print(f'- Files written: {filenames}')
        
        child_ids = get_children(project_name, parent_sample_id)
        child_indices = [ int(child_id.split('-')[-1]) for child_id in child_ids ]
        first_new_child_index = max(child_indices) + 1 if child_indices and max(child_indices) >= 0 else 1
        print(f'Existing children for parent {parent_sample_id}: {child_ids}')
        print(f'First new child index: {first_new_child_index}')

        # For each sample, write the z (a subarray of zs)
        prefix = get_prefix(project_name, parent_sample_id)
        for i in range(n_samples):
          id = f'{prefix}{first_new_child_index + i}'
          filename = f'{base_path}/{project_name}/{id}'

          # zs is a list of 3 tensors, each of shape (n_samples, n_tokens)
          # To write the z for a single sample, we need to take a subarray of each tensor
          z = [ z[i:i+1] for z in zs ]

          t.save(z, f'{filename}.z')
          print(f'Wrote {filename}.z')
          child_ids += [ id ]

        return gr.update(
          choices = get_project_samples(project_name),
          value = child_ids[-1]
        )

      def get_audio(project_name, sample_id):

        global base_path, hps

        filename = f'{base_path}/{project_name}/{sample_id}'

        print(f'Loading {filename}.z')
        z = t.load(f'{filename}.z')
        wav = vqvae.decode(z[2:], start_level=2).cpu().numpy()
        # wav is now of shape (1, sample_length, 1), we want (sample_length,)
        wav = wav[0, :, 0]
        print(f'Generated audio: {wav.shape}')

        return ( hps.sr, wav )

      def pick_sample(project_name, sample_id, preview_just_the_last_n_sec, trim_to_n_sec):

        audio = get_audio(project_name, sample_id)
        wav = audio[1]

        # If trim_to_n_sec is set, trim the audio to that length
        if trim_to_n_sec:
          print(f'Trimming to {trim_to_n_sec} seconds')
          wav = wav[ :int( trim_to_n_sec * hps.sr ) ]
        
        # If the preview_just_the_last_n_sec is set, only show the last n seconds
        if preview_just_the_last_n_sec:
          print(f'Trimming audio to last {preview_just_the_last_n_sec} seconds')
          wav = wav[ int( -1 * preview_just_the_last_n_sec * hps.sr ): ]

        # The audio is a tuple of (sr, wav), where wav is of shape (sample_length,)
        # To plot it, we need to convert it to a list of (x, y) points where x is the time in seconds and y is the amplitude
        x = np.arange(0, len(wav)) / hps.sr
        # Add total length in seconds minus the preview length to the x values
        if preview_just_the_last_n_sec:
          x += ( trim_to_n_sec or len(audio[1]) / hps.sr ) - preview_just_the_last_n_sec
        y = wav
        print(f'Plotting {len(x)} points from {x[0]} to {x[-1]} seconds')

        figure = plt.figure()
        # Set aspect ratio to 10:1
        figure.set_size_inches(20, 2)

        # Remove y axis; make x axis go through y=0          
        ax = plt.gca()
        ax.spines['bottom'].set_position('zero')
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # Set minor x ticks every 0.1 seconds
        ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.1))
        # Move x axis to the foreground
        ax.set_axisbelow(False)

        plt.plot(x, y)
        plt.show()        

        return {
          UI.generated_audio: ( audio[0], wav ),
          UI.audio_waveform: figure,
          UI.preview_just_the_last_n_sec: gr.update(
            maximum = int( len(audio[1]) / audio[0] )
          ),
          UI.trim_to_n_sec: gr.update(
            maximum = int( len(audio[1]) / audio[0] ),
          ),
          UI.go_to_children_button: gr.update(
            visible = len(get_children(project_name, sample_id)) > 0
          ),
          UI.go_to_parent_button: gr.update(
            visible = get_parent(project_name, sample_id) is not None
          )
        }

      def refresh_siblings(project_name, sample_id):
        
        if is_none_ish(sample_id):
          return {
            UI.picked_sample: gr.update( visible = False ),
            UI.sample_box: gr.update( visible = False ),
            UI.generate_first_button: gr.update( visible = True ),
          }

        print(f'Changing current sample to {sample_id}...')
        siblings = get_siblings(project_name, sample_id)
        return {
          UI.picked_sample: gr.update(
            choices = siblings,
            value = sample_id,
            visible = len(siblings) > 1
          ),
          UI.sample_box: gr.update( visible = True ),
          UI.generate_first_button: gr.update( visible = False ),
        }

      UI.sample_tree.change(
        inputs = [ UI.project_name, UI.sample_tree ],
        outputs = [ UI.picked_sample, UI.sample_box, UI.generated_audio, UI.audio_waveform, UI.generate_first_button ],
        fn = refresh_siblings,
        api_name = 'get-siblings'        
      )

      # When the generate button is clicked, generate and update the child samples
      UI.generate_first_button.click(
        inputs = [ UI.project_name, UI.sample_tree, *UI.generation_params ],
        outputs = UI.sample_tree,
        fn = generate,
        api_name = 'generate',
      )

      preview_args = dict(
        inputs = [ UI.project_name, UI.picked_sample, UI.preview_just_the_last_n_sec, UI.trim_to_n_sec ],
        outputs = [ 
          UI.generated_audio, UI.audio_waveform, UI.preview_just_the_last_n_sec, UI.trim_to_n_sec,
          UI.go_to_children_button, UI.go_to_parent_button 
        ],
        fn = pick_sample,
      )

      UI.picked_sample.change(**preview_args, api_name = 'get-audio-preview')

      UI.sample_box.render()

      with UI.sample_box:

        UI.generated_audio.render()
        UI.audio_waveform.render()

        gr.Button(
          value = 'Continue',
          variant = 'primary',
        ).click(
          inputs =  [ UI.project_name, UI.picked_sample, *UI.generation_params ],
          outputs = UI.sample_tree,
          fn = generate,
        )

        gr.Button(
          value = 'Try again',          
        ).click(
          inputs = [ UI.project_name, UI.picked_sample, *UI.generation_params ],
          outputs = UI.sample_tree,
          fn = lambda project_name, sample_id, *args: generate(project_name, get_parent(project_name, sample_id), *args),
        )

        UI.go_to_parent_button.render()
        UI.go_to_parent_button.click(
          inputs = [ UI.project_name, UI.picked_sample ],
          outputs = UI.sample_tree,
          fn = get_parent
        )

        UI.go_to_children_button.render()
        UI.go_to_children_button.click(
          inputs = [ UI.project_name, UI.picked_sample ], 
          outputs = UI.sample_tree,
          fn = lambda project_name, sample_id: get_children(project_name, sample_id)[0]
        )

        def delete_sample(project_name, sample_id, confirm):

          if not confirm:
            return {}
          
          # New child sample is the one that goes after the deleted sample
          siblings = get_siblings(project_name, sample_id)
          new_sibling_to_use = siblings[ siblings.index(sample_id) + 1 ] if sample_id != siblings[-1] else None

          # Remove the to-be-deleted sample from the list of child samples
          siblings.remove(sample_id)

          # Delete the sample
          filename = f'{base_path}/{project_name}/{sample_id}'

          for extension in [ '.z', '.wav' ]:
            if os.path.isfile(f'{filename}{extension}'):
              os.remove(f'{filename}{extension}')
              print(f'Deleted {filename}{extension}')
            else:
              print(f'No {filename}{extension} found')
          return {
            UI.picked_sample: gr.update(
              choices = siblings,
              value = new_sibling_to_use,
            ),
            UI.sample_box: gr.update(
              visible = len(siblings) > 0
            ),            
          }
        
        gr.Button('Delete').click(
          inputs = [ UI.project_name, UI.picked_sample, gr.Checkbox(visible=False) ],
          outputs = [ UI.picked_sample, UI.sample_box ],
          fn = delete_sample,
          _js = """
            ( project_name, child_sample_id ) => {
              if ( confirm('Are you sure? There is no undo!') ) {
                return [ project_name, child_sample_id, true ]
              } else {
                throw new Error('Cancelled; not deleting')
              }
            }
          """,
          api_name = 'delete-sample'
        )

        with gr.Accordion( 'Advanced', open = False ):

          UI.preview_just_the_last_n_sec.render()
          UI.preview_just_the_last_n_sec.change(**preview_args)

          UI.trim_to_n_sec.render()
          UI.trim_to_n_sec.change(**preview_args)

          # Also make the cut button visible or not depending on whether the cut value is 0
          UI.trim_to_n_sec.change(
            inputs = UI.trim_to_n_sec,
            outputs = UI.trim_button,
            fn = lambda trim_to_n_sec: gr.update( visible = trim_to_n_sec > 0 )
          )

          UI.trim_button.render()

          def trim(project_name, sample_id, n_sec):

            filename = f'{base_path}/{project_name}/{sample_id}.z'
            print(f'Loading {filename}...')
            z = t.load(filename)
            print(f'Loaded z, z[2] shape is {z[2].shape}')
            n_tokens = seconds_to_tokens(n_sec)
            print(f'Trimming to {n_tokens} tokens')
            z[2] = z[2][:, :n_tokens]
            print(f'z[2].shape = {z[2].shape}')
            t.save(z, filename)
            print(f'Saved z to {filename}')
            return 0
          
          UI.trim_button.click(
            inputs = [ UI.project_name, UI.picked_sample, UI.trim_to_n_sec ],
            outputs = UI.trim_to_n_sec,
            fn = trim,
            api_name = 'trim'
          )       


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