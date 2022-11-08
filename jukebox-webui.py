#@title Jukebox Web UI

#@markdown This Notebook allows you to creating music with OpenAI’s Jukebox model using a simple, web-based UI that uses your Colab Notebook as a backend.
#@markdown I strongly suggest that you refer to the [getting started page](https://github.com/vzakharov/jukebox-webui/blob/main/docs/getting-started.md) before running it.
#@markdown ***

#@markdown ## Parameters
#@markdown ### *Song duration in seconds*
total_duration = 200 #@param {type:"slider", min:60, max:300, step:10}
#@markdown This is the only generation parameter you need to set in advance (instead of setting it in the UI later), as changing the duration requires reloading the model. If you do want to do this, stop the cell and run it again with the new value.
#@markdown

#@markdown ### *Google Drive or Colab’s (non-persistent!) storage*
use_google_drive = True #@param{type:'boolean'}
#@markdown Uncheck if you want to store data locally (or in your Colab instance) instead of Google Drive. Note that in this case your data will be lost when the Colab instance is stopped.
#@markdown

#@markdown ### *Path for projects*
base_path = '/content/drive/My Drive/jukebox-webui' #@param{type:'string'}
#@markdown This is where your projects will go. ```/content/drive/My Drive/``` refers to the very top of your Google Drive. The folder will be automatically created if it doesn’t exist, so you don’t need to create it manually.
#@markdown

#@markdown ### *Path for models*
models_path = '/content/drive/My Drive/jukebox-webui/_data' #@param{type:'string'}
#@markdown This is where your models will be stored. This app is capable of loading the model from an arbitrary path, so storing it on Google Drive will save you the hassle (and time) of having to download or copy it every time you start the instance. The models will be downloaded automatically if they don’t exist, so you don’t need to download them manually.

share_gradio = True #param{type:'boolean'}
# ☝️ Here and below, change #param to #@param if you want to be able to edit the value from the notebook interface. All of these are for advanced uses (and users), so don’t bother with them unless you know what you’re doing.

#@markdown ---
#@markdown That’s it, you can now run the cell. Note that the first time you run it, it will take a few minutes to download the model. Afterwards, re-running the cell will be much faster.

debug_gradio = True #param{type:'boolean'}

reload_all = False #param{type:'boolean'}

# If running locally, comment out the whole try-except block below, otherwise the !-prefixed commands will give a compile-time error (i.e. it will fail even if the corresponding code is not executed). Note that the app was never tested locally (tbh, I didn’t even succeed installing Jukebox on my machine), so it’s not guaranteed to work.
try:

  !nvidia-smi
  assert not reload_all
  repeated_run
  # ^ If this doesn't give an error, it means we're in Colab and re-running the notebook (because repeated_run is defined in the first run)
  print('Re-running the notebook')

except:
  
  if use_google_drive:
    from google.colab import drive
    drive.mount('/content/drive')

  !nvidia-smi
  !pip install git+https://github.com/openai/jukebox.git
  !pip install gradio

  repeated_run = True
 

# import glob
from datetime import datetime
import random
import gradio as gr
import librosa
import os
import re
# from matplotlib import pyplot as plt
import numpy as np
import torch as t
import urllib.request
# import uuid
import yaml

import jukebox
import jukebox.utils.dist_adapter as dist

from jukebox.make_models import make_vqvae, make_prior, MODELS
from jukebox.hparams import Hyperparams, setup_hparams, REMOTE_PREFIX
from jukebox.utils.dist_utils import setup_dist_from_mpi
from jukebox.utils.remote_utils import download
from jukebox.utils.torch_utils import empty_cache
from jukebox.sample import sample_partial_window, load_prompts

### Model

raw_to_tokens = 128
chunk_size = 16
lower_batch_size = 16
lower_level_chunk_size = 32

hps = Hyperparams()
hps.sr = 44100
hps.levels = 3
hps.hop_fraction = [ 0.5, 0.5, 0.125 ]
hps.sample_length = int(total_duration * hps.sr // raw_to_tokens) * raw_to_tokens

reload_dist = False #param{type:'boolean'}

try:
  assert not reload_dist and not reload_all
  rank, local_rank, device
  print('Dist already setup')
except:
  rank, local_rank, device = setup_dist_from_mpi()
  print(f'Dist setup: rank={rank}, local_rank={local_rank}, device={device}')

# Monkey patch jukebox.make_models.load_checkpoint to load cached checkpoints from local_data_path instead of '~/.cache'
reload_monkey_patch = False #param{type:'boolean'}
try:
  assert not reload_monkey_patch and not reload_all
  monkey_patched_load_checkpoint, monkey_patched_load_audio
  print('Jukebox methods already monkey patched.')
except:

  print('Monkey patching Jukebox methods...')

  try:
    monkey_patched_load_checkpoint
    print('load_checkpoint already monkey patched.')
  except:
    # Monkey patch load_checkpoint, allowing to load models from arbitrary paths
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
      global models_path
      restore = path
      if restore.startswith(REMOTE_PREFIX):
          remote_path = restore
          local_path = os.path.join(models_path, remote_path[len(REMOTE_PREFIX):])
          if dist.get_rank() % 8 == 0:
              download_to_cache(remote_path, local_path)
          restore = local_path
      dist.barrier()
      checkpoint = t.load(restore, map_location=t.device('cpu'))
      print("Restored from {}".format(restore))
      return checkpoint

    jukebox.make_models.load_checkpoint = monkey_patched_load_checkpoint
    print('load_checkpoint monkey patched.')

    # # Download jukebox/models/5b/vqvae.pth.tar and jukebox/models/5b_lyrics/prior_level_2.pth.tar right away to avoid downloading them on the first run
    # for model_path in ['jukebox/models/5b/vqvae.pth.tar', 'jukebox/models/5b_lyrics/prior_level_2.pth.tar']:
    #   download_to_cache(f'{REMOTE_PREFIX}{model_path}', os.path.join(data_path, model_path))

  try:
    monkey_patched_load_audio
    print('load_audio already monkey patched.')
  except:

    # Monkey patch load_audio, allowing for duration = None
    def monkey_patched_load_audio(file, sr, offset, duration, mono=False):
      # Librosa loads more filetypes than soundfile
      x, _ = librosa.load(file, sr=sr, mono=mono, offset=offset/sr, duration=None if duration is None else duration/sr)
      if len(x.shape) == 1:
          x = x.reshape((1, -1))
      return x

    jukebox.utils.audio_utils.load_audio = monkey_patched_load_audio
    print('load_audio monkey patched.')

  print('Monkey patching done.')

reload_prior = False #param{type:'boolean'}

try:
  vqvae, priors, top_prior
  assert total_duration == calculated_duration and not reload_prior and not reload_all
  print('Model already loaded.')
except:

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

  vqvae, *priors = MODELS['5b_lyrics']

  vqvae = make_vqvae(setup_hparams(vqvae, dict(sample_length = hps.sample_length)), device)

  top_prior = make_prior(setup_hparams(priors[-1], dict()), vqvae, device)

  calculated_duration = total_duration


# If the base folder doesn't exist, create it
if not os.path.isdir(base_path):
  os.makedirs(base_path)

try:
  calculated_metas
  print('Using calculated metas')
except:
  calculated_metas = {}

loaded_settings = {}
custom_parents = None

class UI:

  ### Meta

  separate_tab_warning = gr.Box(
    visible = False
  )

  separate_tab_link = gr.Textbox(
    visible = False
  )

  main_window = gr.Row(
    visible = False
  )

  ### General

  project_name = gr.Dropdown(
    label = 'Project'
  )

  create_project_box = gr.Box(
    visible = False
  )

  new_project_name = gr.Textbox(
    label = 'Project name',
    placeholder = 'lowercase-digits-and-dashes-only'
  )

  settings_box = gr.Accordion(
    label = "Settings",
    visible = False
  )

  general_settings = [ project_name ]

  ### Project-specific

  ## Metas (artist, genre, lyrics)
  artist = gr.Dropdown(
    label = 'Artist'
  )

  genre = gr.Dropdown(
    label = 'Genre'
  )

  lyrics = gr.Textbox(
    label = 'Lyrics (optional)',
    max_lines = 5,
    placeholder = 'Shift+Enter for a new line'
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

  getting_started_column = gr.Column( scale = 2, elem_id = 'getting-started-column' )
  
  workspace_column = gr.Column( scale = 3, visible = False )

  primed_audio = gr.Audio(
    label = 'Audio to start from (optional)',
    source = 'microphone'
  )

  # Virtual timestamp textbox to do certain things once the audio is primed (and this textbox is updated), accurate to the millisecond
  prime_timestamp = gr.Textbox(
    value = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
    visible = False
  )

  first_generation_row = gr.Row(
    visible = False
  )

  generation_progress = gr.Markdown('', elem_id = 'generation-progress')

  routed_sample_id = gr.State()

  sample_tree_row = gr.Row(
    visible = False
  )

  sample_tree = gr.Dropdown(
    label = 'Sample tree',
  )

  show_leafs_only = gr.Checkbox(
    label = 'Do not show intermediate samples',
  )


  picked_sample = gr.Radio(
    label = 'Sibling samples',
  )

  sample_box = gr.Box(
    visible = False
  )

  generated_audio = gr.Audio(
    label = 'Generated audio',
    elem_id = "generated-audio"
  )

  audio_waveform = gr.HTML(
    elem_id = 'audio-waveform'
  )

  audio_timeline = gr.HTML(
    elem_id = 'audio-timeline'
  )

  go_to_parent_button = gr.Button(
    value = 'To parent generation',
  )

  go_to_children_button = gr.Button(
    value = 'To child generations',
  )

  total_audio_length = gr.Number(
    label = 'Total audio length, sec',
    elem_id = 'total-audio-length'
  )

  preview_just_the_last_n_sec = gr.Number(
    label = 'Preview just the last ... seconds (0 to disable)'
  )

  trim_to_n_sec = gr.Number(
    label = 'Trim to ... seconds (0 to disable)'
  )

  trim_button = gr.Button( 'Trim', visible = False )

  project_settings = [ *generation_params, sample_tree, show_leafs_only, preview_just_the_last_n_sec ]

  input_names = { input: name for name, input in locals().items() if isinstance(input, gr.components.FormComponent) }

  inputs_by_name = { name: input for name, input in locals().items() if isinstance(input, gr.components.FormComponent) }

def convert_name(name):
  return re.sub(r'[^a-z0-9]+', '-', name.lower())

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

def convert_audio_to_sample(project_name, audio, sec_to_trim_primed_audio, show_leafs_only):

  global hps, base_path

  # audio is of the following form:
  # (48000, array([        0,         0,         0, ..., -26209718, -25768554,       -25400996], dtype=int32))
  # dtype=int16 is also possible
  # Or, if it is a stereo file, audio[1] is [[left, right], [left, right], ...]
  
  print(f'Audio: {audio}')
  # breakpoint()
  print(f'Audio length: {len(audio[1])} samples, sample rate: {audio[0]} Hz')

  # If it is a stereo file, we need to convert it to mono by averaging the left and right channels
  if len(audio[1].shape) > 1:
    audio = (audio[0], audio[1].mean(axis=1))
    print(f'Converted stereo to mono (shape: {audio[1].shape})')

  if sec_to_trim_primed_audio:
    audio = (audio[0], audio[1][:int(audio[0] * sec_to_trim_primed_audio)])
    print(f'Trimmed audio to {sec_to_trim_primed_audio} seconds')

  # Convert the audio to float depending on the dtype

  x = audio[1] / 2**31 if audio[1].dtype == np.int32 else audio[1] / 2**15
  print(f'Converted to [-1, 1]; min = {x.min()}, max = {x.max()}')

  # Resample the audio to hps.sr (if needed)

  if audio[0] != hps.sr:
    x = librosa.resample(x, audio[0], hps.sr)
    print(f'Resampled audio to {hps.sr}')

  # Convert the audio to a tensor (e.g. from array([[-1.407e-03, -4.461e-04, ..., -3.042e-05,  1.277e-05]], dtype=float32) to tensor([[-1.407e-03], [-4.461e-04], ..., [-3.042e-05], [ 1.277e-05]], dtype=float32))

  if len(x.shape) == 1:
    x = x.reshape((1, -1))
    print(f'Reshaped audio to {x.shape}')

  x = x.T
  print(f'Transposed audio to {x.shape}')
  
  xs = [ x ]

  print(f'Created {len(xs)} samples of {x.shape} shape each')

  x = t.stack([t.from_numpy(x) for x in xs])
  print(f'Stacked samples to {x.shape}')

  x = x.to(device, non_blocking=True)
  print(f'Moved samples to {device}')

  zs = top_prior.encode( x, start_level=0, end_level=len(priors), bs_chunks=x.shape[0] )
  print(f'Encoded audio to zs of shape {[ z.shape for z in zs ]}')

  primed_sample_id = f'{project_name}-{get_first_free_index(project_name)}'
  filename = f'{base_path}/{project_name}/{primed_sample_id}.z'
  t.save(zs, filename)

  return {
    UI.sample_tree: gr.update(
      choices = get_samples(project_name, show_leafs_only),
      value = primed_sample_id
    ),
    UI.prime_timestamp: datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
  }

def delete_sample(project_name, sample_id, confirm):

  if not confirm:
    return {}
  
  # New child sample is the one that goes after the deleted sample
  siblings = get_siblings(project_name, sample_id)
  current_index = siblings.index(sample_id)
  new_sibling_to_use = siblings[ current_index + 1 ] if current_index < len(siblings) - 1 else siblings[ current_index - 1 ]

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

def generate(project_name, parent_sample_id, show_leafs_only, artist, genre, lyrics, n_samples, temperature, generation_length):

  print(f'Generating {n_samples} sample(s) of {generation_length} sec each for project {project_name}...')

  global total_duration
  global calculated_metas
  global hps, raw_to_tokens, chunk_size, lower_batch_size, lower_level_chunk_size
  global top_prior, device, priors
  global metas, labels

  hps.n_samples = n_samples

  # If metas or n_samples have changed, recalculate the metas
  if calculated_metas != dict( artist = artist, genre = genre, lyrics = lyrics ) or len(metas) != n_samples:

    print(f'Metas or n_samples have changed, recalculating the model for {artist}, {genre}, {lyrics}, {n_samples} samples...')

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

  if parent_sample_id:

    zs = t.load(f'{base_path}/{project_name}/{parent_sample_id}.z')
    print(f'Loaded parent sample {parent_sample_id} of shape {[ z.shape for z in zs ]}')
    zs = [ z[0].repeat(n_samples, 1) for z in zs ]
    print(f'Converted to shape {[ z.shape for z in zs ]}')

  else:
    zs = [ t.zeros(n_samples, 0, dtype=t.long, device='cuda') for _ in range(3) ]
    print('No parent sample or primer provided, starting from scratch')
  
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

  prefix = get_prefix(project_name, parent_sample_id)
  # For each sample, write the z (a subarray of zs)

  try:
    first_new_child_index = get_first_free_index(project_name, parent_sample_id)
  except Exception as e:
    print(f'Something went wrong: {e}')
    first_new_child_index = random.randrange(1e6, 1e7)
    print(f'Using random index {first_new_child_index} as a fallback')

  for i in range(n_samples):
    id = f'{prefix}{first_new_child_index + i}'
    filename = f'{base_path}/{project_name}/{id}'

    # zs is a list of 3 tensors, each of shape (n_samples, n_tokens)
    # To write the z for a single sample, we need to take a subarray of each tensor
    z = [ z[i:i+1] for z in zs ]

    t.save(z, f'{filename}.z')
    print(f'Wrote {filename}.z')

  return {
    UI.sample_tree: gr.update(
      choices = get_samples(project_name, show_leafs_only),
      value = id
    ),
    UI.generation_progress: f'Generation completed at {datetime.now().strftime("%H:%M:%S")}'
  }

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

def get_children(project_name, parent_sample_id, include_custom=False):

  global base_path

  prefix = get_prefix(project_name, parent_sample_id)
  child_ids = []
  for filename in os.listdir(f'{base_path}/{project_name}'):
    match = re.match(f'{prefix}(\d+)\\.zs?', filename)
    if match:
      child_ids += [ filename.split('.')[0] ]
    
  if include_custom:

    custom_parents = get_custom_parents(project_name)

    for sample_id in custom_parents:
      if custom_parents[sample_id] == parent_sample_id:
        child_ids += [ sample_id ]        

  # print(f'Children of {parent_sample_id}: {child_ids}')

  return child_ids

def get_custom_parents(project_name):

  global base_path, custom_parents
  
  if not custom_parents or custom_parents['project_name'] != project_name:
    print('Loading custom parents...')
    custom_parents = {}
    filename = f'{base_path}/{project_name}/{project_name}-parents.yaml'
    if os.path.exists(filename):
      print(f'Found {filename}')
      with open(filename) as f:
        loaded_dict = yaml.load(f, Loader=yaml.FullLoader)
        print(f'Loaded as {loaded_dict}')
        # Add project_name to the beginning of every key and value in the dictionary
        custom_parents = { f'{project_name}-{k}': f'{project_name}-{v}' for k, v in loaded_dict.items() }

    custom_parents['project_name'] = project_name
    
    print(f'Custom parents: {custom_parents}')

  return custom_parents

def get_first_free_index(project_name, parent_sample_id = None):
  print(f'Getting first free index for {project_name}, parent {parent_sample_id}')
  child_ids = get_children(project_name, parent_sample_id, include_custom=False)
  print(f'Child ids: {child_ids}')
  # child_indices = [ int(child_id.split('-')[-1]) for child_id in child_ids ]
  child_indices = []
  for child_id in child_ids:
    suffix = child_id.split('-')[-1]
    # If not an integer, ignore
    if suffix.isdigit():
      child_indices += [ int(suffix) ]
  
  first_free_index = max(child_indices) + 1 if child_indices and max(child_indices) >= 0 else 1
  print(f'First free index: {first_free_index}')

  return first_free_index

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
    UI.genre: gr.update(
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
    )
  }

lists = {}
def get_list(what):
  items = []
  # print(f'Getting {what} list...')
  # If list already exists, return it
  if what in lists:
    # print(f'{what} list already exists.')
    return lists[what]
  else:
    with urllib.request.urlopen(f'https://raw.githubusercontent.com/openai/jukebox/master/jukebox/data/ids/v2_{what}_ids.txt') as f:
      for line in f:
        item = line.decode('utf-8').split(';')[0]
        item = item.replace('_', ' ').title()
        items.append(item)
    items.sort()
    print(f'Loaded {len(items)} {what}s.')
    lists[what] = items
  return items

def get_parent(project_name, sample_id):

  global base_path
  
  custom_parents = get_custom_parents(project_name)

  if sample_id in custom_parents:
    return custom_parents[sample_id]

  # Remove the project name and first dash from the sample id
  path = sample_id[ len(project_name) + 1: ].split('-')
  parent_sample_id = '-'.join([ project_name, *path[:-1] ]) if len(path) > 1 else None
  # print(f'Parent of {sample_id}: {parent_sample_id}')
  return parent_sample_id

def get_prefix(project_name, parent_sample_id):
  return f'{parent_sample_id or project_name}-'

def get_samples(project_name, show_leafs_only):

  choices = []
  for filename in os.listdir(f'{base_path}/{project_name}'):
    if re.match(r'.*\.zs?$', filename):
      id = filename.split('.')[0]
      if show_leafs_only and len( get_children(project_name, id) ) > 0:
        continue
      choices += [ id ]
  
  # Sort by id, in descending order
  choices.sort(reverse = True)
  
  return choices

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

def get_siblings(project_name, sample_id):

  return get_children(project_name, get_parent(project_name, sample_id))

def is_new(project_name):
  return project_name == 'CREATE NEW' or not project_name
  
def get_project(project_name, routed_sample_id):

  global base_path, loaded_settings

  is_this_new = is_new(project_name)

  # Start with default values for project settings
  settings_out_dict = {
    UI.artist: 'Unknown',
    UI.genre: 'Unknown',
    UI.lyrics: '',
    UI.generation_length: 1,
    UI.temperature: 0.98,
    UI.n_samples: 2,
    UI.sample_tree: None
  }

  samples = []
  sample = None

  # If not new, load the settings from settings.yaml in the project folder, if it exists
  if not is_this_new:

    print(f'Loading settings for {project_name}...')

    settings_path = f'{base_path}/{project_name}/{project_name}.yaml'
    if os.path.isfile(settings_path):
      with open(settings_path, 'r') as f:
        loaded_settings = yaml.load(f, Loader=yaml.FullLoader)
        print(f'Loaded settings for {project_name}: {loaded_settings}')

        # Go through all the settings and set the value for settings_out_dict where the key is the element itself
        for key, value in loaded_settings.items():
          if key in UI.inputs_by_name and UI.inputs_by_name[key] in UI.project_settings:

            input = UI.inputs_by_name[key]

            # If the value is an integer (i) but the element is an instance of gr.components.Radio or gr.components.Dropdown, take the i-th item from the choices
            if isinstance(value, int) and isinstance(input, (gr.components.Radio, gr.components.Dropdown)):
              print(f'Converting {key} value {value} to {input.choices[value]}')
              value = input.choices[value]
            
            settings_out_dict[getattr(UI, key)] = value

          else:
            print(f'Warning: {key} is not a valid project setting')

    # Write the last project name to settings.yaml
    with open(f'{base_path}/settings.yaml', 'w') as f:
      print(f'Saving {project_name} as last project...')
      yaml.dump({'last_project': project_name}, f)
      print('Saved to settings.yaml')
    
    settings_out_dict[ UI.getting_started_column ] = gr.update(
      visible = False
    )

    samples = get_samples(project_name, settings_out_dict[ UI.show_leafs_only ] if UI.show_leafs_only in settings_out_dict else False)

    sample = routed_sample_id or (
      (
        settings_out_dict[ UI.sample_tree ] or samples[0] 
      ) if len(samples) > 0 else None
    )

    settings_out_dict[ UI.sample_tree ] = gr.update(
      choices = samples,
      value = sample 
    )


  return {
    UI.create_project_box: gr.update( visible = is_this_new ),
    UI.settings_box: gr.update( visible = not is_this_new ),
    UI.workspace_column: gr.update( visible = not is_this_new  ),
    UI.sample_box: gr.update( visible = sample is not None ),
    UI.first_generation_row: gr.update( visible = len(samples) == 0 ),
    UI.sample_tree_row: gr.update( visible = len(samples) > 0 ),
    **settings_out_dict
  }

def get_sample(project_name, sample_id, preview_just_the_last_n_sec, trim_to_n_sec):

  audio = get_audio(project_name, sample_id)
  wav = audio[1]

  # If trim_to_n_sec is set, trim the audio to that length
  if trim_to_n_sec:
    print(f'Trimming to {trim_to_n_sec} seconds')
    wav = wav[ :int( trim_to_n_sec * hps.sr ) ]
  
  # If the preview_just_the_last_n_sec is set, only show the last n seconds
  if preview_just_the_last_n_sec:
    print(f'Trimming audio to last {preview_just_the_last_n_sec} seconds')
    # As the audio length can be non-integer, add its fractional part to the preview length
    preview_just_the_last_n_sec += ( len(wav) / hps.sr ) % 1
    wav = wav[ int( -1 * preview_just_the_last_n_sec * hps.sr ): ]

  return {
    UI.generated_audio: ( audio[0], wav ),
    UI.total_audio_length: int( len(audio[1]) / hps.sr * 100 ) / 100,
    UI.go_to_children_button: gr.update(
      visible = len(get_children(project_name, sample_id)) > 0
    ),
    UI.go_to_parent_button: gr.update(
      visible = get_parent(project_name, sample_id) is not None
    ),
    UI.sample_box: gr.update(
      visible = True
    ),
  }

def refresh_siblings(project_name, sample_id):
  
  if not sample_id:
    return {
      UI.picked_sample: gr.update( visible = False )
    }

  print(f'Changing current sample to {sample_id}...')
  siblings = get_siblings(project_name, sample_id)
  return gr.update(
    choices = siblings,
    value = sample_id,
    visible = len(siblings) > 1
  )

def rename_sample(project_name, old_sample_id, new_sample_id, show_leafs_only):

  if not re.match(r'^[a-zA-Z0-9-]+$', new_sample_id):
    raise ValueError('Sample ID must be alphanumeric and dashes only')

  new_sample_id = f'{project_name}-{new_sample_id}'

  print(f'Renaming {old_sample_id} to {new_sample_id}')

  custom_parents = get_custom_parents(project_name)
  print(f'Custom parents: {custom_parents}')
  custom_parents[new_sample_id] = get_parent(project_name, old_sample_id)
  print(f'Added {new_sample_id} -> {custom_parents[new_sample_id]} to custom parents')
  if old_sample_id in custom_parents:
    del custom_parents[old_sample_id]
    print(f'Removed {old_sample_id} from custom parents')

  # Find all samples that have this sample as a custom parent and update them
  for child, parent in custom_parents.items():
    if parent == old_sample_id:
      custom_parents[child] = new_sample_id
      print(f'Updated {child} -> {new_sample_id} in custom parents')
  
  print(f'Final custom parents: {custom_parents}')
  
  # Save the new custom parents
  with open(f'{base_path}/{project_name}/{project_name}-parents.yaml', 'w') as f:
    # Dump everything but the "project_name" key and remove the "project_name-" prefix
    custom_parents_to_save = {
      k[len(project_name)+1:]: v[len(project_name)+1:] for k, v in custom_parents.items() if k != 'project_name'
    }
    print(f'Writing: {custom_parents_to_save}')
    yaml.dump(custom_parents_to_save, f)
    print('Done.')

  # Find all files in the project directory that start with the old sample ID followed by either a dash or a dot and rename them
  for filename in os.listdir(f'{base_path}/{project_name}'):
    if re.match(rf'^{old_sample_id}[-.]', filename):
      new_filename = filename.replace(old_sample_id, new_sample_id)
      print(f'Renaming {filename} to {new_filename}')
      os.rename(f'{base_path}/{project_name}/{filename}', f'{base_path}/{project_name}/{new_filename}')
    
  return gr.update(
    choices = get_samples(project_name, show_leafs_only),
    value = new_sample_id
  )

def save_project(project_name, *project_input_values):

  if is_new(project_name):
    return

  # print(f'Saving settings for {project_name}...')
  # print(f'Project input values: {project_input_values}')

  # Go through all UI attributes and add the ones that are in the project settings to a dictionary
  settings = {}

  for i in range(len(UI.project_settings)):
    settings[UI.input_names[UI.project_settings[i]]] = project_input_values[i]
  
  # print(f'Settings: {settings}')

  # If the settings are different from the loaded settings, save them to the project folder

  if settings != loaded_settings:

    with open(f'{base_path}/{project_name}/{project_name}.yaml', 'w') as f:
      yaml.dump(settings, f)
      print(f'Saved settings to {base_path}/{project_name}/{project_name}.yaml: {settings}')
  
  # else:
  #   print('Settings are the same as loaded settings, not saving.')

def seconds_to_tokens(sec):

  global hps, raw_to_tokens, chunk_size

  tokens = sec * hps.sr // raw_to_tokens
  tokens = ( (tokens // chunk_size) + 1 ) * chunk_size
  return int(tokens)

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

with gr.Blocks(
  css = """
    .gr-button {
      /* add margin to the button */
      margin: 5px 5px 5px 5px;
    }

    #getting-started-column {
      /* add a considerable margin to the left of the column */
      margin-left: 20px;
    }

    #generation-progress {
      /* gray, smaller font */
      color: #888;
      font-size: 0.8rem;
    }

  """,
  title = 'Jukebox Web UI',
) as app:

  with UI.separate_tab_warning.render():

    UI.separate_tab_link.render()

    gr.Button('Click here to open the UI', variant = 'primary' ).click( inputs = UI.separate_tab_link, outputs = None, fn = None,
      _js = "link => window.open(link, '_blank')"
    )
  
  with UI.main_window.render():

    with gr.Column( scale = 1 ):

      UI.project_name.render().change(
        inputs = [ UI.project_name, UI.routed_sample_id ],
        outputs = [ 
          UI.create_project_box, UI.settings_box, *UI.project_settings, UI.getting_started_column, UI.workspace_column, UI.first_generation_row, 
          UI.sample_tree_row, UI.sample_box 
        ],
        fn = get_project,
        api_name = 'get-project'
      )

      with UI.create_project_box.render():

        UI.new_project_name.render().blur(
          inputs = UI.new_project_name,
          outputs = UI.new_project_name,
          fn = convert_name,
        )

        # When a project is created, create a subfolder for it and update the project list.
        create_args = dict(
          inputs = UI.new_project_name,
          outputs = UI.project_name,
          fn = create_project,
        )

        UI.new_project_name.submit( **create_args )
        gr.Button('Create project').click( **create_args )

      with UI.settings_box.render():

        for component in UI.generation_params:
          
          # For artist, also add a search button and a randomize button
          if component == UI.artist:

            with gr.Row():

              component.render()
             
              def filter_artists(filter):
                
                artists = get_list('artist')

                if filter:
                  artists = [ artist for artist in artists if filter.lower() in artist.lower() ]
                  artist = artists[0]
                else:
                  # random artist
                  artist = random.choice(artists)

                return gr.update(
                  choices = artists,
                  value = artist
                )

              artist_filter = gr.Textbox(
                label = '🔍',
                placeholder = 'Empty for 🎲',
              )

              artist_filter.submit(
                inputs = artist_filter,
                outputs = UI.artist,
                fn = filter_artists,
                api_name = 'filter-artists'
              )
          
          else:

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
            fn = save_project,
          )


    with UI.getting_started_column.render():

      # Load the getting started text from github (vzakharov/jukebox-webui/docs/getting-started.md) via urllib
      with urllib.request.urlopen('https://raw.githubusercontent.com/vzakharov/jukebox-webui/main/docs/getting-started.md') as f:
        getting_started_text = f.read().decode('utf-8')
        gr.Markdown(getting_started_text)

    with UI.workspace_column.render():

      with gr.Tab('Compose'):

        with gr.Column():

          with UI.first_generation_row.render():

            with gr.Column():
            
              gr.Markdown("""
                To start composing, you need to generate the first batch of samples. You can:
                
                - Start from scratch by clicking the **Generate initial samples** button below, or
                - Go to the **Prime** tab and convert your own audio to a sample.
              """)

              gr.Button('Generate initial samples', variant = "primary" ).click(
                inputs = [ UI.project_name, UI.sample_tree, UI.show_leafs_only, *UI.generation_params ],
                outputs = [ UI.sample_tree, UI.first_generation_row, UI.sample_tree_row, UI.generation_progress ],
                fn = lambda *args: {
                  **generate(*args),
                  UI.first_generation_row: gr.update( visible = False ),
                  UI.sample_tree_row: gr.update( visible = True ),
                }
              )

          with UI.sample_tree_row.render():
            
            UI.routed_sample_id.render()
            UI.sample_tree.render()
            UI.show_leafs_only.render()
            
            UI.show_leafs_only.change(
              inputs = [ UI.project_name, UI.show_leafs_only ],
              outputs = UI.sample_tree,
              fn = lambda *args: gr.update( choices = get_samples(*args) ),
            )

          UI.picked_sample.render()

          UI.sample_tree.change(
            inputs = [ UI.project_name, UI.sample_tree ],
            outputs = UI.picked_sample,
            fn = refresh_siblings,
            api_name = 'get-siblings'        
          )

          preview_args = dict(
            inputs = [ UI.project_name, UI.picked_sample, UI.preview_just_the_last_n_sec, UI.trim_to_n_sec ],
            outputs = [ 
              UI.sample_box, UI.generated_audio, UI.total_audio_length, 
              UI.go_to_children_button, UI.go_to_parent_button
            ],
            fn = get_sample,
          )

          UI.picked_sample.change(
            **preview_args,
            api_name = 'get-sample',
            _js =
            # Set the search string to ?[sample_id] for easier navigation
            '''
              ( ...args ) => (
                args[1] && window.history.pushState( {}, '', `?${args[1]}` ),
                args
              ) 
            '''
          )

          with UI.sample_box.render():

            for this in [ 
              UI.generated_audio, 
              UI.audio_waveform,
              UI.audio_timeline
            ]:
              this.render()

            gr.HTML("""
              <!-- Button to play/pause the audio -->
              <button class="gr-button gr-button-lg gr-button-secondary"
                onclick = "
                  wavesurfer.playPause()
                  this.innerText = wavesurfer.isPlaying() ? '⏸️' : '▶️'
                "
              >▶️</button>

              <!-- Textbox showing current time -->
              <input type="number" class="gr-box gr-input gr-text-input" id="audio-time" value="0" readonly>
            """)


            gr.Button(
              value = 'Generate further',
              variant = 'primary',
            ).click(
              inputs =  [ UI.project_name, UI.picked_sample, UI.show_leafs_only, *UI.generation_params ],
              outputs = [ UI.sample_tree, UI.generation_progress ],
              fn = generate,
            )

            gr.Button(
              value = 'Generate more variations',          
            ).click(
              inputs = [ UI.project_name, UI.picked_sample, UI.show_leafs_only, *UI.generation_params ],
              outputs = [ UI.sample_tree, UI.generation_progress ],
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

              with gr.Tab('Manipulate audio'):

                UI.total_audio_length.render()
                UI.preview_just_the_last_n_sec.render().blur(**preview_args)
                UI.trim_to_n_sec.render().blur(**preview_args)

                # Also make the cut button visible or not depending on whether the cut value is 0
                UI.trim_to_n_sec.change(
                  inputs = UI.trim_to_n_sec,
                  outputs = UI.trim_button,
                  fn = lambda trim_to_n_sec: gr.update( visible = trim_to_n_sec )
                )

                UI.trim_button.render()

                UI.trim_button.click(
                  inputs = [ UI.project_name, UI.picked_sample, UI.trim_to_n_sec ],
                  outputs = UI.trim_to_n_sec,
                  fn = trim,
                  api_name = 'trim'
                )
              
              with gr.Tab('Rename sample'):

                new_sample_id = gr.Textbox(
                  label = 'New sample id',
                  placeholder = 'Alphanumeric and dashes only'
                )

                gr.Button('Rename').click(
                  inputs = [ UI.project_name, UI.picked_sample, new_sample_id, UI.show_leafs_only ],
                  outputs = UI.sample_tree,
                  fn = rename_sample,
                  api_name = 'rename-sample'
                )

        UI.generation_progress.render()

      with gr.Tab('Prime'):

        primed_audio_source = gr.Radio(
          label = 'Audio source',
          choices = [ 'microphone', 'upload' ],
          value = 'microphone'
        )

        UI.primed_audio.render()
        
        primed_audio_source.change(
          inputs = primed_audio_source,
          outputs = UI.primed_audio,
          fn = lambda source: gr.update( source = source ),
        )

        sec_to_trim_primed_audio = gr.Number(
          label = 'Trim starting audio to ... seconds from the beginning',
        )

        def trim_primed_audio(audio, sec):
          print(f'Trimming {audio} to {sec} seconds')
          # # Plot the audio to console for debugging
          # plt.plot(audio)
          # plt.show()              
          # Audio is of the form (sr, audio)
          trimmed_audio = audio[1][:int(sec * audio[0])]
          print(f'Trimmed audio shape is {trimmed_audio.shape}')
          return ( audio[0], trimmed_audio )

        sec_to_trim_primed_audio.submit(
          inputs = [ UI.primed_audio, sec_to_trim_primed_audio ],
          outputs = UI.primed_audio,
          fn = trim_primed_audio
        )

        prime_button = gr.Button(
          'Convert to sample',
          variant = 'primary'
        )
                
        prime_button.click(
          inputs = [ UI.project_name, UI.primed_audio, sec_to_trim_primed_audio, UI.show_leafs_only ],
          outputs = [ UI.sample_tree, prime_button, UI.prime_timestamp ], # UI.prime_timestamp is updated to the current time to force tab change
          fn = convert_audio_to_sample,
          api_name = 'convert-wav-to-sample'
        )

        UI.prime_timestamp.render().change(
          inputs = UI.prime_timestamp, outputs = None, fn = None,
          _js = 
            # Find a button inside a div inside another div with class 'tabs', the button having 'Compose' as text, and click it -- all this in the shadow DOM.
            # Gosh, this is ugly.
            """
              timestamp => {
                console.log(`Timestamp changed to ${timestamp}; clicking the 'Compose' tab`)
                for ( let button of document.querySelector('gradio-app').shadowRoot.querySelectorAll('div.tabs > div > button') ) {
                  if ( button.innerText == 'Compose' ) {
                    button.click()
                    break
                  }
                }
                return timestamp
              }
            """
        )


  def dummy_string_input():
    return gr.Textbox( visible = False )
    
  app.load(
    on_load,
    inputs = [ gr.Textbox(visible=False), gr.Textbox(visible=False), gr.Textbox(visible=False) ],
    outputs = [ UI.project_name, UI.routed_sample_id, UI.artist, UI.genre, UI.getting_started_column, UI.separate_tab_warning, UI.separate_tab_link, UI.main_window ],
    api_name = 'initialize',
    _js = """async (...args) => {
      
      try {

        // Create and inject wavesurfer scripts
        let require = url => {
          console.log(`Injecting ${url}`)
          let script = document.createElement('script')
          script.src = url
          document.head.appendChild(script)
          return new Promise( resolve => script.onload = () => {
            console.log(`Injected ${url}`)
            resolve()
          } )
        }

        await require('https://cdnjs.cloudflare.com/ajax/libs/wavesurfer.js/6.3.0/wavesurfer.min.js')
        await require('https://cdnjs.cloudflare.com/ajax/libs/wavesurfer.js/6.3.0/plugin/wavesurfer.timeline.min.js')

        // The wavesurfer element is hidden inside a shadow DOM hosted by <gradio-app>, so we need to get it from there
        let shadowSelector = selector => document.querySelector('gradio-app').shadowRoot.querySelector(selector)

        let waveformDiv = shadowSelector('#audio-waveform')
        console.log(`Found waveform div:`, waveformDiv)

        let timelineDiv = shadowSelector('#audio-timeline')
        console.log(`Found timeline div:`, timelineDiv)
        
        let getAudioTime = time => {
          let previewDuration = wavesurfer.getDuration()
          // Take total duration from #total-audio-length's input
          let totalDuration = parseFloat(shadowSelector('#total-audio-length input').value)
          let additionalDuration = totalDuration - previewDuration
          return Math.round( ( time + additionalDuration ) * 100 ) / 100          
        }

        // Create a (global) wavesurfer object with and attach it to the div
        window.wavesurfer = WaveSurfer.create({
          container: waveformDiv,
          waveColor: 'skyblue',
          progressColor: 'steelblue',
          plugins: [
            WaveSurfer.timeline.create({
              container: timelineDiv,
              // Light colors, as the background is dark
              primaryColor: '#eee',
              secondaryColor: '#ccc',
              primaryFontColor: '#eee',
              secondaryFontColor: '#ccc',
              formatTimeCallback: time => Math.round(getAudioTime(time))
            })
          ]
        })
        
        // Add a seek event listener to the wavesurfer object, modifying the #audio-time input
        wavesurfer.on('seek', progress => {
          shadowSelector('#audio-time').value = getAudioTime(progress * wavesurfer.getDuration())
        })

        // Also update the time when the audio is playing
        wavesurfer.on('audioprocess', time => {
          shadowSelector('#audio-time').value = getAudioTime(time)
        })

        // Put an observer on #generated-audio (also in the shadow DOM) to reload the audio from its inner <audio> element
        let parentElement = document.querySelector('gradio-app').shadowRoot.querySelector('#generated-audio')
        let parentObserver

        parentObserver = new MutationObserver( mutations => {
          // Check if there is an inner <audio> element
          let audioElement = parentElement.querySelector('audio')
          if ( audioElement ) {
            
            console.log('Found audio element:', audioElement)

            // If so, create an observer on it while removing the observer on the parent
            parentObserver.disconnect()

            let audioSrc

            let reloadAudio = () => {
              // Check if the audio element has a src attribute and if it has changed
              if ( audioElement.src && audioElement.src !== audioSrc ) {
                // If so, reload the audio
                audioSrc = audioElement.src
                console.log('Reloading audio from', audioSrc)
                wavesurfer.load(audioSrc)
              }
            }

            reloadAudio()

            let audioObserver = new MutationObserver( reloadAudio )

            audioObserver.observe(audioElement, { attributes: true })

          }

        })

        parentObserver.observe(parentElement, { childList: true, subtree: true })

        // href, query_string, error_message
        return [ window.location.href, window.location.search.slice(1), null ]

      } catch (e) {

        console.error(e)

        // If anything went wrong, return the error message
        return [ window.location.href, window.location.search.slice(1), e.message ]

      }
    }"""
  )

  app.launch( share = share_gradio, debug = debug_gradio )