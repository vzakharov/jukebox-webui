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

metas = None
labels = None

def calculate_model(duration, duration_used_by_model, artist, genre, lyrics, n_samples):

  global hps, raw_to_tokens, chunk_size
  global vqvae, priors, top_prior, device
  global metas, labels

  if duration != duration_used_by_model:

    print(f'Duration {duration} is not equal to duration used by model {duration_used_by_model}; recalculating vqvae and priors...')

    hps.sample_length = int(duration * hps.sr // raw_to_tokens) * raw_to_tokens

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

  return {
    UI.duration_used_by_model: duration
  }


def seconds_to_tokens(sec):

  global hps, raw_to_tokens, chunk_size

  tokens = sec * hps.sr // raw_to_tokens
  tokens = ( (tokens // chunk_size) + 1 ) * chunk_size
  return int(tokens)

def generate(base_folder, project_name, prime_file, parent_sample_id, generation_length, trim_prime_seconds, n_samples, temperature):

  global hps, raw_to_tokens, chunk_size, lower_batch_size, lower_level_chunk_size
  global top_prior

  sample_hps = Hyperparams(dict(
    mode = 'primed' if prime_file else 'ancestral',
    codes_file = None,
    audio_file = prime_file,
    prompt_length_in_seconds = trim_prime_seconds,
  ))

  if prime_file:
    trim_prime_samples = int( trim_prime_seconds * hps.sr // raw_to_tokens ) * raw_to_tokens
    wav = load_prompts([ prime_file], trim_prime_samples, hps)
    zs = top_prior.encode(wav, start_level=0, end_level=len(priors), bs_chunks=wav.shape[0])
  
  else:
    zs = [ t.zeros(hps.n_samples, 0, dtype=t.long, device='cuda') for _ in range(3) ]
  
  zs = [ z.repeat(n_samples, 1) for z in zs ]

  tokens_to_sample = seconds_to_tokens(generation_length)
  sampling_kwargs = dict(
    temp=temperature, fp16=True, max_batch_size=lower_batch_size,
    chunk_size=lower_level_chunk_size
  )

  zs = sample_partial_window(zs, labels, sampling_kwargs, 2, top_prior, tokens_to_sample, hps)
  wav = vqvae.decode(zs[2:], start_level=2).cpu().numpy()

  write_files(base_folder, project_name, zs, wav, parent_sample_id)


def write_files(base_folder, project_name, zs, wav, parent_sample_id=''):

  # 1. Scan project folder for [project_name][-parent_sample_id]-[comma-separated child ids].zs
  # 2. Take the highest child id and add 1 to it (call it first_new_child_id)
  # 3. Create [project_name][-parent_sample_id]-[first_new_child_id,first_new_child_id+1,first_new_child_id+2,etc.].zs depending on how many samples we have (based on the shape of zs)

  global hps
  n_samples = wav.shape[0]

  base_filename = f'{base_folder}/{project_name}/{project_name}'
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


### UX

def get_project_names(base_folder):
  project_names = []
  for folder in os.listdir(base_folder):
    if os.path.isdir(base_folder+'/'+folder) and not folder.startswith('_'):
      project_names.append(folder)
  # Sort project names alphabetically
  project_names.sort()
  # Add "CREATE NEW" option in the beginning
  return ['CREATE NEW'] + project_names

def get_list(name):
  items = []
  with urllib.request.urlopen(f'https://raw.githubusercontent.com/openai/jukebox/master/jukebox/data/ids/v2_{name}_ids.txt') as f:
    for line in f:
      item = line.decode('utf-8').split(';')[0]
      item = item.replace('_', ' ').title()
      items.append(item)
  items.sort()
  return items

def get_files(path, extension):
  files = []
  for file in os.listdir(path):
    if file.endswith(extension):
      files.append(file)
  # Sort backwards
  files.sort(reverse=True)
  return files

def update_samples(base_folder, project_name):
  print(f'Updating samples list for {project_name}...')
  # If the filename is 'project-part1-1-3-2-4,5,6.zs', sample ids will be 'part1-1-3-2-4', 'part1-1-3-2-5', and 'part1-1-3-2-6'
  sample_ids = []
  for file in os.listdir(f'{base_folder}/{project_name}'):
    if file.endswith('.zs'):
      # Remove path, project name, and extension
      full_id = file.replace(f'{project_name}-', '').replace('.zs', '')
      parts = full_id.split('-')
      sample_ids += [ 
        '-'.join(parts[:-1] + [id]) 
          for id in parts[-1].split(',') 
      ]
  # Reverse sort
  sample_ids.sort(reverse=True)
  return gr.update(
    choices = sample_ids,
  )

def update_primes(base_folder, project_name):
  print(f'Updating prime wavs list for {project_name}...')
  # Search for files formatted as [project name]-prime[-optional suffix].wav
  prime_wavs = get_files(f'{base_folder}/{project_name}', '.wav')
  prime_wavs = [wav for wav in prime_wavs if re.match(f'^{project_name}-prime(-.*)?\.wav$', wav)]
  return {
    UI.prime_file: gr.update(
      choices = prime_wavs,
      visible = len(prime_wavs) > 0,
    ),
    UI.prime_tip: gr.update(
      visible = len(prime_wavs) == 0,
    ),
  }


def update_project_data(base_folder, project_name):

  # By default, take values from the inputs in the project-specific settings
  out_dict = { input: input.value for input in UI.project_specific_inputs }

  # Load data from settings.yaml in the project folder, if it exists
  data_filename = f'{base_folder}/{project_name}/settings.yaml'
  if os.path.exists(data_filename):
    with open(data_filename, 'r') as f:
      data = yaml.load(f, Loader=yaml.FullLoader)
      print(f'Loaded settings from {data_filename}:')
      print(data)
      for key in data:
        out_dict[getattr(UI, key)] = data[key]

  return out_dict

def save_project_data(base_folder, project_name, artist, genre, lyrics, duration, duration_used_by_model, step, sample_id, use_prime, prime_file, generation_length):
  # Dump all arguments except base_folder and project_name
  data = {key: value for key, value in locals().items() if key not in ['base_folder', 'project_name']}
  filename = f'{base_folder}/{project_name}/settings.yaml'
  with open(filename, 'w') as f:
    yaml.dump(data, f)
    # print(f'Settings updated.')
    # print(data) 

class UI:

  # Define UI elements

  base_folder = gr.Textbox(label='Base folder')
  project_name = gr.Dropdown(label='Project name')

  artist = gr.Dropdown(label='Artist', choices=get_list('artist'))
  genre = gr.Dropdown(label='Genre', choices=get_list('genre'))
  lyrics = gr.Textbox(label='Lyrics', max_lines=8)

  duration = gr.Slider(label='Duration', minimum=60, maximum=600, step=10)
  duration_used_by_model = gr.Number(visible=False)

  step = gr.Radio(label='Step', choices=['Start', 'Continue'])

  use_prime = gr.Checkbox(label='Start with an existing audio sample')
  prime_file = gr.Dropdown(label='Prime filename')
  prime_tip = gr.Markdown('To start with an existing audio, upload it to the project folder, name it as `[project name]-prime[-optional suffix].wav` and refresh this page.', visible=False)

  sample_id = gr.Dropdown(label='Sample Id')

  generation_length = gr.Slider(label='Generation length', minimum=1, maximum=10, step=0.25)

  sample_audio = gr.Audio(visible=False)

  sample_children_audios = [ gr.Audio() for i in range(10) ]

  all_inputs = [ input for input in locals().values() if isinstance(input, gr.components.IOComponent) ]

  project_defining_inputs = [ base_folder, project_name ]
  project_specific_inputs = [ artist, genre, lyrics, duration ]
  generation_specific_inputs = [ step, sample_id, use_prime, prime_file, generation_length ]

  general_inputs = project_defining_inputs

  with gr.Blocks() as ui:

    with gr.Row():

      with gr.Column(scale=1):

        with gr.Box():
          
          base_folder.render()

          # Radio buttons to either open an existing project or create a new one
          open_or_create_project = gr.Radio(
            label='Project',
            choices=['Open', 'New'],
            value='Open'
          )

          with gr.Box(visible=True) as open_project_box:

            # Open a project
            project_name.render()
          
          with gr.Box(visible=False) as create_project_box:
            # Create a new project

            new_project_name = gr.Textbox(label='New project name')

            def create_new_project(base_folder, new_project_name):
              print(f'Creating new project {new_project_name}...')
              path = f'{base_folder}/{new_project_name}'
              # If the project folder already exists, throw an error
              assert not os.path.exists(path), f'Project folder {path} already exists!'
              os.mkdir(f'{base_folder}/{new_project_name}')
              return {
                UI.project_name: gr.update(
                  choices = get_project_names(base_folder),
                  value = new_project_name,
                ),
                UI.open_project_box: gr.update(visible=True),
                UI.create_project_box: gr.update(visible=False),
                UI.open_or_create_project: 'Open'
              }
            
            create_project_button = gr.Button('Create')

            create_project_args = {
              'inputs': [ base_folder, new_project_name ],
              'outputs': [ project_name, open_project_box, create_project_box, open_or_create_project ],
              'fn': create_new_project
            }

            create_project_button.click(**create_project_args)
            new_project_name.submit(**create_project_args)

          open_or_create_project.change(
            inputs = open_or_create_project,
            outputs = [ open_project_box, create_project_box ],
            fn = lambda action: [
              gr.update(visible=a==action) for a in ['Open', 'New']
            ]
          )
        
        with gr.Box(visible=False) as project_box:
          
          [ input.render() for input in project_specific_inputs ]
        
          # If duration changes and is not equal to the duration used by the model, show a button to recalculate the model

          caculate_model_button = gr.Button(visible=False).click(
            inputs = duration,
            outputs = duration_used_by_model,
            fn = calculate_model
          )

          duration.change(
            inputs = [ duration, duration_used_by_model ],
            outputs = caculate_model_button,
            fn = lambda duration, duration_used_by_model: gr.update(
              visible=duration!=duration_used_by_model,
              label='Recalculate model' if duration_used_by_model else 'Calculate model'
            )
          )

      with gr.Column(scale=3, visible=False) as generation_box:

        step.render()

        with gr.Box(visible=False) as start_box:

          use_prime.render()

          with gr.Box(visible=False) as prime_box:

            prime_file.render()
            prime_tip.render()
          
          use_prime.change(
            inputs = use_prime,
            outputs = prime_box,
            fn = lambda use_prime: gr.update(visible=use_prime)
          )

        with gr.Box(visible=False) as continue_box:
          sample_id.render()

          # Button to go to parent sample
          def go_to_parent(base_folder, project_name, sample_id):
            parent_id = re.sub(r'-\d+$', '', sample_id)
            assert parent_id != sample_id, 'Can’t deduce parent id from sample id'
            # Assert that parent id exists (there is a wav file for it)
            assert os.path.exists(f'{base_folder}/{project_name}/{project_name}-{parent_id}.wav'), 'Parent id does not exist'
            return parent_id

          gr.Button('<< To parent sample').click(
            go_to_parent,
            inputs = [ base_folder, project_name, sample_id ],
            outputs = sample_id,
          )
          
        generation_length.render()

        generate_button = gr.Button('Generate')

        sample_audio.render()

        with gr.Box(visible=False) as children_box:

          gr.Markdown('**Children**')

          child_boxes = []
          child_ids = []

          for i in range(10):

            with gr.Box(visible=False) as child_box:

              sample_children_audios[i].render()

              child_id = gr.Textbox(visible=False)

              gr.Button('Go to').click(
                inputs = child_id,
                outputs = sample_id,
                fn = lambda child_id: child_id
              )

              child_boxes += [ child_box ]
              child_ids += [ child_id ]

    # Event logic

    # If base_folder changes, update available project names
    base_folder.submit(
      lambda base_folder: gr.update(choices=get_project_names(base_folder)),
      inputs=base_folder,
      outputs=project_name,
    )

    def update_project(base_folder, project_name):

      print(f'Loading project {project_name}...')

      # If project_name is 'CREATE NEW', show the create project box and hide the project/generation boxes
      if project_name == 'CREATE NEW':
        return {
          UI.create_project_box: gr.update(visible=True),
          UI.project_box: gr.update(visible=False),
          UI.generation_box: gr.update(visible=False),
        }

      else:

        # Write general data to settings.json
        print('Saving general data')
        data = {}
        items = UI.__dict__.items()
        for i, input in enumerate(UI.general_inputs):
          print(f'Input {i}: {input}')
          name = [ key for key, value in items if value == input ][0]
          value = [ base_folder, project_name ][i]
          print(f'{name}: {value}')
          data[name] = value
        with open('settings.json', 'w') as f:
          json.dump(data, f)

        return {
          UI.project_box: gr.update(visible=True),
          UI.generation_box: gr.update(visible=True),
          **update_project_data(base_folder, project_name),
          **update_primes(base_folder, project_name),
          UI.sample_id: update_samples(base_folder, project_name),
        }

    project_name.change(
      inputs = general_inputs,
      outputs = all_inputs + [ project_box, generation_box ],
      fn = update_project
    )

    # Whenever a project-specific input changes, save the project data
    for input in project_specific_inputs + generation_specific_inputs:
      input.change(save_project_data, 
        inputs = project_defining_inputs + project_specific_inputs + generation_specific_inputs,
        outputs = [],
      )
    
    step.change(
      lambda step: [ gr.update(visible=step==current_step) for current_step in ['Start', 'Continue'] ],
      inputs=step, outputs=[start_box, continue_box]
    )      

    # When the sample changes, unhide if [project_name]-[sample_id].wav exists and point it to that file
    def update_sample_wavs(base_folder, project_name, sample_id):
      filename = f'{base_folder}/{project_name}/{project_name}-{sample_id}.wav'
      print(f'Looking for {filename}')

      if os.path.exists(filename):

        print(f'Found {filename}')
        child_audios = []
        child_ids = []
        visibilities = []

        for file in glob.glob(f'{base_folder}/{project_name}/{project_name}-{sample_id}-*.wav'):

          # Remove project name and extension
          match = re.match(f'.*{project_name}-{sample_id}-(\d+)\.wav', file)
          if not match:
            continue
          child_id = match.group(1)
          if child_id:
            print(f'Found child {child_id}')
            visibilities += [ gr.update(visible=True) ]
            child_audios += [ 
              gr.update(
                value=file,
                label=f'{sample_id}-{child_id}',
              )
            ]
            child_ids += [ f'{sample_id}-{child_id}' ]
            # If already at max children, stop
            if len(child_audios) == 10:
              break
        
        # If length is less than 10, hide the rest
        for i in range(len(child_audios), 10):
          child_audios += [ gr.update() ]
          visibilities += [ gr.update(visible=False) ]
          child_ids += [ gr.update() ]
        
        # Hide the entire children box if there are no children
        children_box_visible = gr.update(visible=len(child_audios) > 0)

        return [ children_box_visible, gr.update(visible=True, value=filename, label=sample_id) ] + child_audios + child_ids + visibilities

      else:

        return [ gr.update(visible=False) ] * 2 + [ gr.update() for i in range(20) ] + [ gr.update(visible=False) for i in range(10) ]
    
    sample_id.change(update_sample_wavs,
      inputs = project_defining_inputs + [sample_id],
      outputs = [ children_box, sample_audio ] + sample_children_audios + child_ids + child_boxes,
    )

    # On load, load general data from server
    def load_general_data(*args):
      print('Loading general data')
      # Convert general inputs to a dict so we can return it
      out = {}
      for i, value in enumerate(args):
        print(f'Loading {i} = {value}')
        out[UI.general_inputs[i]] = value
      print(f'General data: {out}')
      # Read 'settings.json' from root folder if it exists
      if os.path.exists('settings.json'):
        print('Found settings.json')
        with open('settings.json', encoding='utf-8') as f:
          # For every key, find the corresponding input and set its value
          for key, value in json.load(f).items():
            out[UI.__dict__[key]] = value
      else:
        print('No settings.json found')
      
      print(f'General data: {out}')

      return out
    
    ui.load(load_general_data, inputs=general_inputs, outputs=general_inputs)



  def __init__(self):
    self.ui.launch(share=True, debug=True)

  # Set default values directly
  base_folder.value = '/content/drive/MyDrive/AI music'

  project_name.value = ''
  # project_name.choices = get_project_names(base_folder.value)

  artist.value = 'Unknown'
  genre.value = 'Unknown'
  lyrics.value = ''
  # duration.value = 200

  # step.value = 'Start'

  sample_id.value = ''
  prime_file.value = ''

  generation_length.value = 3

if __name__ == '__main__':
  UI()