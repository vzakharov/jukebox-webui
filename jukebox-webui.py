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

import subprocess

# If running locally, comment out the whole try-except block below, otherwise the !-prefixed commands will give a compile-time error (i.e. it will fail even if the corresponding code is not executed). Note that the app was never tested locally (tbh, I didn’t even succeed installing Jukebox on my machine), so it’s not guaranteed to work.

try:

  !nvidia-smi
  empty_cache()
  print('Cache cleared.')
  !nvidia-smi
  assert not reload_all
  repeated_run
  # ^ If this doesn't give an error, it means we're in Colab and re-running the notebook (because repeated_run is defined in the first run)
  print('Re-running the notebook')

except:
  
  if use_google_drive:
    from google.colab import drive
    drive.mount('/content/drive')

  !pip install git+https://github.com/openai/jukebox.git
  !pip install gradio

  repeated_run = True
 

# import glob
import base64
from datetime import datetime
import hashlib
import random
import shutil
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
from jukebox.utils.sample_utils import get_starts
from jukebox.utils.torch_utils import empty_cache
from jukebox.sample import sample_partial_window, load_prompts, upsample, sample_single_window

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


class Upsampling:

  started = False
  zs = None
  project = None
  sample_id = None
  level = None
  metas = None
  labels = None
  priors = None
  params = None

  windows = []
  window_index = 0
  elapsed_time = None
  time_per_window = None
  windows_remaining = None
  time_remaining = None

print('Monkey patching Jukebox methods...')

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

# Monkey patch load_audio, allowing for duration = None
def monkey_patched_load_audio(file, sr, offset, duration, mono=False):
  # Librosa loads more filetypes than soundfile
  x, _ = librosa.load(file, sr=sr, mono=mono, offset=offset/sr, duration=None if duration is None else duration/sr)
  if len(x.shape) == 1:
      x = x.reshape((1, -1))
  return x

jukebox.utils.audio_utils.load_audio = monkey_patched_load_audio
print('load_audio monkey patched.')


# Monkey patch sample_level, saving the current upsampled z to respective file
# The original code is as follows:
# def sample_level(zs, labels, sampling_kwargs, level, prior, total_length, hop_length, hps):
#   print_once(f"Sampling level {level}")
#   if total_length >= prior.n_ctx:
#       for start in get_starts(total_length, prior.n_ctx, hop_length):
#           zs = sample_single_window(zs, labels, sampling_kwargs, level, prior, start, hps)
#   else:
#       zs = sample_partial_window(zs, labels, sampling_kwargs, level, prior, total_length, hps)
#   return zs
#
# Rewritten:

def monkey_patched_sample_level(zs, labels, sampling_kwargs, level, prior, total_length, hop_length, hps):

  if not Upsampling.started:
    # We stopped upsampling in the UI, so now we need to reload the top prior (as we deleted it before upsampling) and exit
    load_top_prior()
    return zs

  global base_path

  # The original code provides for shorter samples by sampling only a partial window, but we'll just throw an error for simplicity
  assert total_length >= prior.n_ctx, f'Total length {total_length} is shorter than prior.n_ctx {prior.n_ctx}'

  Upsampling.zs = zs
  Upsampling.level = level

  print(f"Sampling level {level}")
  # Remember current time
  start_time = datetime.now()
  Upsampling.windows = get_starts(total_length, prior.n_ctx, hop_length)

  print(f'Totally {len(Upsampling.windows)} windows at level {level}')

  # Remove all windows whose start + n_ctx is less than however many samples we've already upsampled (at this level)
  already_upsampled = Upsampling.zs[level].shape[1]
  if already_upsampled > 0:
    print(f'Already upsampled {already_upsampled} samples at level {level}')
    Upsampling.windows = [ start for start in Upsampling.windows if start + prior.n_ctx > already_upsampled ]

  if len(Upsampling.windows) == 0:
    print(f'No windows to upsample at level {level}')
  else:
    print(f'Upsampling {len(Upsampling.windows)} windows, from {Upsampling.windows[0]} to {Upsampling.windows[-1]+prior.n_ctx}')

    Upsampling.window_index = 0
    for start in Upsampling.windows:

      # If we stopped upsampling in the UI, stop here
      if not Upsampling.started:
        load_top_prior()
        return zs

      window_start_time = datetime.now()

      print(f'Sampling window starting at {start}')
      Upsampling.zs = sample_single_window(Upsampling.zs, labels, sampling_kwargs, level, prior, start, hps)

      # Estimate time remaining
      Upsampling.elapsed_time = datetime.now() - start_time
      Upsampling.time_per_window = datetime.now() - window_start_time
      Upsampling.windows_remaining = len(Upsampling.windows) - Upsampling.window_index
      Upsampling.time_remaining = Upsampling.time_per_window * Upsampling.windows_remaining
      print(f'Elapsed time: {Upsampling.elapsed_time}, time remaining for level {level}: {Upsampling.time_remaining}')

      path = f'{base_path}/{Upsampling.project}/{Upsampling.sample_id}.z'
      print(f'Saving upsampled z to {path}')
      t.save(Upsampling.zs, path)
      print('Done.')
      Upsampling.window_index += 1

  return Upsampling.zs

jukebox.sample.sample_level = monkey_patched_sample_level
print('sample_level monkey patched.')

reload_prior = False #param{type:'boolean'}

def load_top_prior():
  global top_prior, vqvae, device

  print('Loading top prior')
  top_prior = make_prior(setup_hparams(priors[-1], dict()), vqvae, device)


if Upsampling.started:
  print('''
    !!! APP SET FOR UPSAMPLING !!!

    To use the app for composing, stop execution, create a new cell and run the following code:

    Upsampling.started = False

    Then run the main cell again.
  ''')
else:

  try:
    vqvae, priors, top_prior

    assert total_duration == calculated_duration and not reload_prior and not reload_all
    print('Model already loaded.')
  except:

    print(f'Loading vqvae and top_prior for duration {total_duration}...')

    vqvae, *priors = MODELS['5b_lyrics']

    vqvae = make_vqvae(setup_hparams(vqvae, dict(sample_length = hps.sample_length)), device)

    load_top_prior()

    calculated_duration = total_duration

    empty_cache


# If the base folder doesn't exist, create it
if not os.path.isdir(base_path):
  os.makedirs(base_path)

try:
  calculated_metas
  print('Calculated metas already loaded.')
except:
  calculated_metas = {}
  print('Calculated metas created.')

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

  generation_progress = gr.Markdown('Generation status will be shown here', elem_id = 'generation-progress')

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
    label = 'Variations',
  )

  sample_box = gr.Box(
    visible = False
  )

  upsampling_row = gr.Row(
    visible = False
  )

  UPSAMPLING_LEVEL_NAMES = [ 'Original', 'Midsampled', 'Upsampled' ]

  upsampling_level = gr.Radio(
    label = 'Upsampling level',
    choices = [ 'Original' ],
    value = 'Original'
  )

  upsample_rendering = gr.Radio(
    label = 'Render...',
    type = 'index',
    choices = [ 'Channel 1', 'Channel 2', 'Channel 3', 'Pseudo-stereo', 'Pseudo-stereo with delay' ],
    value = 'Pseudo-stereo with delay',
  )

  generated_audio = gr.Audio(
    label = 'Generated audio',
    elem_id = 'generated-audio',
  )

  mp3_file = gr.File(
    label = 'MP3',
    elem_id = 'audio-file',
    type = 'binary',
    visible = False
  )
  
  audio_waveform = gr.HTML(
    elem_id = 'audio-waveform'
  )

  audio_timeline = gr.HTML(
    elem_id = 'audio-timeline'
  )

  compose_row = gr.Box(
    elem_id = 'compose-row',
  )

  go_to_parent_button = gr.Button(
    value = '<',
  )

  go_to_children_button = gr.Button(
    value = '>',
  )

  total_audio_length = gr.Number(
    label = 'Total audio length, sec',
    elem_id = 'total-audio-length',
    interactive = False,
  )

  preview_just_the_last_n_sec = gr.Number(
    label = 'Preview just the last ... seconds (0 to disable)'
  )

  trim_to_n_sec = gr.Number(
    label = 'Trim to ... seconds (0 to disable)',
    elem_id = 'trim-to-n-sec'
  )

  trim_button = gr.Button( 'Trim', visible = False )

  sample_to_upsample = gr.Textbox(
    label = 'Sample to upsample',
    placeholder = 'Choose a sample in the Compose tab first',
    interactive = False,
  )

  genre_for_upsampling_left_channel = gr.Dropdown(
    label = 'Genre for upsampling (left channel)'
  )

  genre_for_upsampling_center_channel = gr.Dropdown(
    label = 'Genre for upsampling (center channel)'
  )

  genre_for_upsampling_right_channel = gr.Dropdown(
    label = 'Genre for upsampling (right channel)'
  )

  upsample_button = gr.Button('Start upsampling', variant="primary", elem_id='upsample-button')

  upsampling_tracker_markdown = gr.Markdown('')

  upsampling_in_progress = gr.Number(0, visible = False)
  # Note: for some reason, Gradio doesn't monitor programmatic changes to a checkbox, so we use a number instead

  project_settings = [ 
    *generation_params, sample_tree, show_leafs_only, preview_just_the_last_n_sec,
    genre_for_upsampling_left_channel, genre_for_upsampling_center_channel, genre_for_upsampling_right_channel 
  ]

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


def get_z(project_name, sample_id):
  global base_path

  return t.load(f'{base_path}/{project_name}/{sample_id}.z')

def get_levels(project_name, sample_id):

  z = get_z(project_name, sample_id)
  
  # z is a list of 3 tensors, each of shape (n_samples, n_tokens). We need to return the levels that have at least one token
  return [ i for i in range(3) if z[i].shape[1] > 0 ]

def get_audio(project_name, sample_id, trim_to_n_sec, preview_just_the_last_n_sec, level=2, upsample_rendering=3):

  print(f'Generating audio for {project_name}/{sample_id} (level {level}, upsample_rendering {upsample_rendering}, trim_to_n_sec {trim_to_n_sec}, preview_just_the_last_n_sec {preview_just_the_last_n_sec})')

  # Get current GPU memory usage. If it's above 12GB, empty the cache
  memory = t.cuda.memory_allocated()
  print(f'GPU memory usage is {memory / 1e9:.1f} GB')
  if t.cuda.memory_allocated() > 12e9:
    print('GPU memory usage is above 12GB, clearing the cache')
    empty_cache()
    print(f'GPU memory usage is now {t.cuda.memory_allocated() / 1e9:1f} GB')

  global base_path, hps

  filename = f'{base_path}/{project_name}/{sample_id}'

  print(f'Loading {filename}.z')
  z = t.load(f'{filename}.z')
  z = z[level]
  # z is of shape torch.Size([1, n_tokens])
  # print(f'Loaded {filename}.z at level {level}, shape: {z.shape}')

  total_audio_length = int( tokens_to_seconds(z.shape[1], level) * 100 ) / 100

  if trim_to_n_sec:
    trim_to_tokens = seconds_to_tokens(trim_to_n_sec, level)
    # print(f'Trimming to {trim_to_n_sec} seconds ({trim_to_tokens} tokens)')
    z = z[ :, :trim_to_tokens ]
    # print(f'Trimmed to shape: {z.shape}')
  
  if preview_just_the_last_n_sec:
    preview_tokens = seconds_to_tokens(preview_just_the_last_n_sec, level)
    # print(f'Trimming audio to last {preview_just_the_last_n_sec} seconds ({preview_tokens} tokens)')
    preview_tokens += ( len(z) / seconds_to_tokens(1, level) ) % 1
    z = z[ :, int( -1 * preview_tokens ): ]
    # print(f'Trimmed to shape: {z.shape}')

  def decode(z):
    wav = vqvae.decode([ z ], start_level=level, end_level=level+1).cpu().numpy()
    # the decoded wav is of shape (n_samples, sample_length, 1)
    return wav
  
  # If z is longer than 30 seconds, there will likely be not enough RAM to decode it in one go
  # In this case, we'll split it into 30-second chunks (with a 1-second overlap), decode each chunk separately, and concatenate the results, crossfading the overlaps
  if z.shape[1] < seconds_to_tokens(30, level):
    wav = decode(z)
  else:
    chunk_size = seconds_to_tokens(30, level)
    overlap_size = seconds_to_tokens(1, level)
    print(f'z is too long ({z.shape[1]} tokens), splitting into chunks of {chunk_size} tokens, with a {overlap_size} token overlap')
    wav = None
    # Keep in mind that the last chunk can be shorter if the total length is not a multiple of chunk_size)
    for i in range(0, z.shape[1], chunk_size - overlap_size):

      # If this is the last chunk, move i back so that the chunk ends at the end of z
      # Also increase the overlap size respectively (so that i + overlap_size is where the previous chunk ended)
      overflow = i + chunk_size - z.shape[1]
      is_last_chunk = overflow > 0
      if is_last_chunk:
        i -= overflow
        overlap_size += overflow

      left_overlap_z = z[ :, i:i+overlap_size ]
      print(f'Left overlap (tokens): {left_overlap_z.shape[1]}')
      left_overlap = decode(left_overlap_z)
      print(f'Left overlap (samples): {left_overlap.shape[1]}')

      # Fade in the left overlap and add it to the existing wav if it's not empty (i.e. if this is not the first chunk)
      if wav is not None:
        left_overlap *= np.linspace(0, 1, left_overlap.shape[1]).reshape(1, -1, 1)
        wav[ :, -left_overlap.shape[1]: ] += left_overlap
        print(f'Added left overlap to wav, overall shape now: {wav.shape}')
      else:
        wav = left_overlap
        print(f'Created wav with left overlap')

      main_chunk_z = z[ :, i+overlap_size:i+chunk_size-overlap_size ]
      print(f'Main chunk (tokens): {main_chunk_z.shape[1]}')

      if main_chunk_z.shape[1] > 0:

        main_chunk = decode(main_chunk_z)
        print(f'Main chunk (samples): {main_chunk.shape[1]}')

        # Add the main chunk to the existing wav
        wav = np.concatenate([ wav, main_chunk ], axis=1)
        print(f'Added main chunk to wav, overall shape now: {wav.shape}')

      else:
        print('Main chunk is empty, skipping')
        continue

      # Fade out the right overlap, unless this is the last chunk
      if not is_last_chunk:

        right_overlap_z = z[ :, i+chunk_size-overlap_size:i+chunk_size ]
        print(f'Right overlap (tokens): {right_overlap_z.shape[1]}')
        right_overlap = decode(right_overlap_z)
        print(f'Right overlap (samples): {right_overlap.shape[1]}')

        right_overlap *= np.linspace(1, 0, right_overlap.shape[1]).reshape(1, -1, 1)

        # Add the right overlap to the existing wav
        wav = np.concatenate([ wav, right_overlap ], axis=1)
        print(f'Added right overlap to wav, overall shape now: {wav.shape}')
      
      else:

        print(f'Last chunk, not adding right overlap')
      
      print(f'Decoded {i+chunk_size} tokens out of {z.shape[1]}, wav shape: {wav.shape}')

  # wav is now of shape (n_samples, sample_length, 1)
  # If this is level 2, we want just (sample_length,), picking the first sample if there are multiple
  if level == 2:
    wav = wav[0, :, 0]

  # Otherwise, this is a batch of upsampled audio, so we need to act depending on the upsample_rendering parameter
  else:

    # upsample_rendering of 0, 1 or 2 means we just need to pick one of the samples
    if upsample_rendering < 3:

      wav = wav[upsample_rendering, :, 0]
    
    # upsample_rendering of 3 means we need to convert the audio to stereo, putting sample 0 to the left, 1 to the center, and 2 to the right
    # 4 means we also want to add a delay of 20 ms for the left and 40 ms for the right channel

    else:

      def to_stereo(wav, stereo_delay_ms=0):

        # A stereo wav is of form (sample_length + double the delay, 2)
        delay_quants = int( stereo_delay_ms * hps.sr / 1000 )
        stereo = np.zeros((wav.shape[1] + 2 * delay_quants, 2))
        # First let's convert the wav to [n_quants, n_samples] by getting rid of the last dimension and transposing the rest
        wav = wav[:, :, 0].T
        print(f'Converted wav to shape {wav.shape}')
        # Take sample 0 for left channel (delayed once), 1 for both channels (non-delayed), and sample 2 for right channel (delayed twice)
        if delay_quants:
          stereo[ delay_quants: -delay_quants, 0 ] = wav[ :, 0 ]
          stereo[ 2 * delay_quants:, 1 ] = wav[ :, 2 ]
          stereo[ : -2 * delay_quants, 0 ] += wav[ :, 1 ]
          stereo[ : -2 * delay_quants, 1 ] += wav[ :, 1 ]
        else:
          stereo[ :, 0 ] = wav[ :, 0 ] + wav[ :, 1 ]
          stereo[ :, 1 ] = wav[ :, 2 ] + wav[ :, 1 ]
        # Now we have max amplitude of 2, so we need to divide by 2
        stereo /= 2

        print(f'Converted to stereo with delay {stereo_delay_ms} ms, current shape: {stereo.shape}, max/min amplitudes: {np.max(stereo)}/{np.min(stereo)}')

        return stereo
      
      wav = to_stereo(wav, stereo_delay_ms=20 if upsample_rendering == 4 else 0)

  print(f'Generated audio of length {len(wav)} ({ len(wav) / hps.sr } seconds); original length: {total_audio_length} seconds.')

  return wav, total_audio_length

# def get_audio_being_upsampled():

#   if not Upsampling.started:
#     return None

#   zs = Upsampling.zs
#   level = Upsampling.level

#   print(f'Generating audio for level {level}')
  
#   x = Upsampling.priors[level].decode(zs[level:], start_level=level, bs_chunks=zs[level].shape[0])

#   # x is of shape (1, sample_length, 1), we want (sample_length,)
#   wav = x[0, :, 0].cpu().numpy()
#   return gr.update(
#     visible = True,
#     value = ( hps.sr, wav ),
#   )

def get_children(project_name, parent_sample_id, include_custom=True):

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
    UI.sample_tree: None,
    UI.genre_for_upsampling_left_channel: 'Unknown',
    UI.genre_for_upsampling_center_channel: 'Unknown',
    UI.genre_for_upsampling_right_channel: 'Unknown',
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

def get_sample(project_name, sample_id, preview_just_the_last_n_sec, trim_to_n_sec, level_name, upsample_rendering):

  global hps

  level_names = UI.UPSAMPLING_LEVEL_NAMES
  level = 2 - level_names.index(level_name)

  print(f'Loading sample {sample_id} for level {level} ({level_name})')

  filename = f'{base_path}/{project_name}/rendered/{sample_id} L{level}'

  # Add trimmed/preview suffixes
  if trim_to_n_sec:
    filename += f' trim {trim_to_n_sec}'
  if preview_just_the_last_n_sec:
    filename += f' preview {preview_just_the_last_n_sec}'
  
  # Add lowercase of upsample rendering option
  if upsample_rendering and level_name != 'Original':
    suffixes = [ 'left', 'center', 'right', 'stereo', 'stereo-delay' ]
    filename += f' {suffixes[upsample_rendering]}'
  
  # Add a hash of the corresponding z file (so that we can detect if the z file has changed and hence we need to re-render)
  filename += f' {hashlib.md5(open(f"{base_path}/{project_name}/{sample_id}.z", "rb").read()).hexdigest()}'

  print(f'Checking if {filename}.wav/.mp3 exist...')

  # If the filenames do not exist, render it
  if not os.path.isfile(f'{filename}.wav') or not os.path.isfile(f'{filename}.mp3'):

    wav, total_audio_length = get_audio(project_name, sample_id, trim_to_n_sec, preview_just_the_last_n_sec, level, upsample_rendering)

    if not os.path.exists(os.path.dirname(filename)):
      os.makedirs(os.path.dirname(filename))

    librosa.output.write_wav(f'{filename}.wav', np.asfortranarray(wav), hps.sr)

    # Convert to mp3
    subprocess.run(['ffmpeg', '-y', '-i', f'{filename}.wav', '-acodec', 'libmp3lame', '-ab', '320k', f'{filename}.mp3'])

  else:
    wav, _ = librosa.load(filename + '.wav', sr=hps.sr)
    total_audio_length = wav.shape[0] / hps.sr
    print(f'Loaded {filename}.wav of shape {wav.shape} ({total_audio_length} sec)')

  return {
    UI.generated_audio: gr.update(
      value = ( hps.sr, wav ),
      label = f'{sample_id} (lossless)',
    ),
    UI.mp3_file: f'{filename}.mp3',
    UI.total_audio_length: total_audio_length,
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

  # print(f'Getting siblings for {sample_id}...')
  siblings = get_siblings(project_name, sample_id)
  # print(f'Siblings for {sample_id}: {siblings}')
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

def seconds_to_tokens(sec, level = 2):

  global hps, raw_to_tokens, chunk_size

  tokens = sec * hps.sr // raw_to_tokens
  tokens = ( (tokens // chunk_size) + 1 ) * chunk_size

  # For levels 1 and 0, multiply by 4 and 16 respectively
  tokens *= 4 ** (2 - level)

  return int(tokens)

def toggle_upsampling(toggle_on, project_name, sample_id, artist, lyrics, *genres):

  global hps, top_prior, priors

  print(f'Toggling upsampling for {sample_id} to {toggle_on}')

  Upsampling.project = project_name
  Upsampling.sample_id = sample_id

  if not toggle_on:
    Upsampling.started = False
    print('Upsampling stopped. You’ll still have to wait for the current window to finish, which may take 5-10 minutes.')
    return

  print(f'Upsampling {sample_id} with genres {genres}')
  filename = f'{base_path}/{project_name}/{sample_id}.z'

  Upsampling.zs = t.load(filename)
  # zs is a list of 3 tensors, one per level, each of shape (n_samples, n_tokens).
  # If the number of samples for any level is other than 3, we need to repeat the first sample to make it 3.
  Upsampling.zs = [ z[0].repeat(3, 1) if z.shape[0] != 3 else z for z in Upsampling.zs ]

  # We also need to create new labels from the metas with the genres replaced accordingly
  Upsampling.metas = [ dict(
    artist = artist,
    genre = genre,
    total_length = hps.sample_length,
    offset = 0,
    lyrics = lyrics,
  ) for genre in genres ]
  
  if not Upsampling.labels:

    try:
      assert top_prior
    except:
      load_top_prior()
    
    Upsampling.labels = top_prior.labeller.get_batch_labels(Upsampling.metas, 'cuda')
    print('Calculated new labels from top prior')

    # We need to delete the top_prior object and empty the cache, otherwise we'll get an OOM error
    del top_prior
    empty_cache()

  labels = Upsampling.labels

  upsamplers = [ make_prior(setup_hparams(prior, dict()), vqvae, 'cpu') for prior in priors[:-1] ]

  if type(labels)==dict:
    labels = [ prior.labeller.get_batch_labels(Upsampling.metas, 'cuda') for prior in upsamplers ] + [ labels ]
    print('Converted labels to list')
    # Not sure why we need to do this -- I copied this from another notebook.
  
  Upsampling.labels = labels
  
  # Create a backup of the original file, in case something goes wrong
  shutil.copy(filename, f'{filename}.bak')
  print(f'Created backup of {filename} as {filename}.bak')

  Upsampling.params = [
    dict(temp=0.99, fp16=True, max_batch_size=16, chunk_size=32),
    dict(temp=0.99, fp16=True, max_batch_size=16, chunk_size=32),
    None
  ]
  
  Upsampling.hps = hps

  # Set hps.n_samples to 3, because we need 3 samples for each level
  Upsampling.hps.n_samples = 3

  # Set hps.sample_length to the actual length of the sample
  Upsampling.hps.sample_length = Upsampling.zs[2].shape[1] * raw_to_tokens

  # Set hps.name to our project directory
  Upsampling.hps.name = f'{base_path}/{project_name}'

  Upsampling.priors = [*upsamplers, None]
  Upsampling.started = True
  Upsampling.zs = upsample(Upsampling.zs, labels, Upsampling.params, Upsampling.priors, Upsampling.hps)

def tokens_to_seconds(tokens, level = 2):

  global hps, raw_to_tokens

  return tokens * raw_to_tokens / hps.sr / 4 ** (2 - level)

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
      color: #777;
      font-size: 0.8rem;
      /* make the height just enough to fit the text */
      height: 1.2rem;
    }

    #audio-timeline {
      /* hide for now */
      display: none;
    }


  """,
  title = 'Jukebox Web UI v0.3',
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
            inputs = [
              UI.project_name, UI.picked_sample, UI.preview_just_the_last_n_sec, UI.trim_to_n_sec, UI.upsampling_level, UI.upsample_rendering
            ],
            outputs = [ 
              UI.sample_box, UI.generated_audio, UI.mp3_file,
              UI.total_audio_length, UI.go_to_children_button, UI.go_to_parent_button,
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

            with UI.upsampling_row.render():

              with gr.Column():

                UI.upsampling_level.render().change(
                  **preview_args,
                )

                # Only show the upsampling elements if there are upsampled versions of the picked sample
                def show_or_hide_upsampling_elements(project_name, sample_id):

                  levels = get_levels(project_name, sample_id)
                  # print(f'Levels: {levels}')

                  available_level_names = UI.UPSAMPLING_LEVEL_NAMES[:len(levels)]
                  # print(f'Available level names: {available_level_names}')

                  return {
                    UI.upsampling_row: gr.update(
                      visible = len(levels) > 1
                    ),
                    UI.upsampling_level: gr.update(
                      choices = available_level_names,
                      # # Choose the highest available level by default.
                      # value = available_level_names[-1]
                    )
                  }
                
                UI.picked_sample.change(
                  inputs = [ UI.project_name, UI.picked_sample ],
                  outputs = [ UI.upsampling_row, UI.upsampling_level ],
                  fn = show_or_hide_upsampling_elements,
                )

              with gr.Column(visible = False) as upsampling_manipulation_row:

                # Show the row only if an upsampled sample is selected and hide the compose row respectively (we can only compose with the original sample)
                UI.upsampling_level.change(
                  inputs = UI.upsampling_level,
                  outputs = [ upsampling_manipulation_row, UI.compose_row ],
                  fn = lambda upsampling_level: [
                    gr.update( visible = upsampling_level != 'Original' ),
                    gr.update( visible = upsampling_level == 'Original' ),
                  ]
                )

                UI.upsample_rendering.render().change(
                  **preview_args,
                )

              continue_upsampling_markdown = gr.Markdown('''
                Upsampling for this sample hasn’t been completed yet. Go to the **Upsample** tab to continue from where it stopped.
              ''', visible = False )

              # Show the continue upsampling markdown only if the current level's length in tokens is less than the total audio length
              # Also update the upsampling button to say "Continue upsampling" instead of "Upsample"
              def show_or_hide_continue_upsampling(project_name, sample_id, total_audio_length):

                z = get_z(project_name, sample_id)[0]

                unfinished = z.shape[1] < seconds_to_tokens(total_audio_length, 0)

                return {
                  continue_upsampling_markdown: gr.update( visible = unfinished ),
                  UI.upsample_button: 'Continue upsampling' if unfinished else 'Upsample',
                }
              
              UI.picked_sample.change(
                inputs = [ UI.project_name, UI.picked_sample, UI.total_audio_length ],
                outputs = [ continue_upsampling_markdown, UI.upsample_button ],
                fn = show_or_hide_continue_upsampling,
              )

            UI.generated_audio.render()

            UI.mp3_file.render()

            # Refresh button
            gr.Button('🔃', elem_id = 'internal-refresh-button', visible=False).click(
              **preview_args,
            )
                
            for element in [ 
              UI.audio_waveform,
              UI.audio_timeline
            ]:
              element.render()


            # Play/pause button, js-based
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

              <!-- Download button -- it will be set to the right href later on -->
              <a class="gr-button gr-button-lg gr-button-secondary" id="download-button">
                🔗
              </a>

              <!-- Refresh button -- it virtually clicks the "internal-refresh-button" button (which is hidden) -->
              <button class="gr-button gr-button-lg gr-button-secondary" onclick="window.shadowRoot.getElementById('internal-refresh-button').click()">
                🔃
              </button>
            """)

            with UI.compose_row.render():

              gr.Button(
                value = 'Go on',
                variant = 'primary',
              ).click(
                inputs =  [ UI.project_name, UI.picked_sample, UI.show_leafs_only, *UI.generation_params ],
                outputs = [ UI.sample_tree, UI.generation_progress ],
                fn = generate,
              )

              gr.Button(
                value = 'More variations',          
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

              gr.Button('🗑️').click(
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

              UI.generation_progress.render()

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

      with gr.Tab('Upsample'):

        # Warning that this process is slow and can take up to 10 minutes for 1 second of audio
        with gr.Accordion('⚠️ WARNING ⚠️', open = False):

          gr.Markdown('''
            Upsampling is a slow process. It can take up to 10 minutes to upsample 1 second of audio in the highest possible quality (2.5 minutes in the medium quality). Currently the Web UI does not show any progress, so consult with your Colab's output to see the progress.

            YOU WILL NOT BE ABLE TO CONTINUE COMPOSING WHILE THE PROCESS IS RUNNING. If you want to continue composing, you can either wait for the process to finish, or you can create a copy of the notebook and start a second Colab session (provided your Colab subscription allows it).

            Once the process is done, you will find the high- and medium-quality samples in the `level_0` and `level_1` folders inside your project folder, respectively.
            
            Note: The tool will be creating intermediate files in the `tmp` folder inside your project folder, so that the work done is not lost. At the same time, there is yet no way to continue an interrupted upsample process, so you’ll need some Python knowledge to do that.
          ''')

        UI.sample_to_upsample.render()

        # Change the sample to upsample when a sample is picked
        UI.picked_sample.change(
          inputs = UI.picked_sample,
          outputs = UI.sample_to_upsample,
          fn = lambda x: x,
        )

        with gr.Accordion('Genres for upsampling (optional)', open = False):

          with gr.Accordion('What is this?', open = False):

            gr.Markdown('''
              The tool will generate three upsamplings of the selected sample, which will then be panned to the left, center, and right, respectively. Choosing different genres for each of the three upsamplings will result in a more diverse sound between them, thus enhancing the (pseudo-)stereo effect. 

              A good starting point is to have a genre that emphasizes vocals (e.g. `Pop`) for the center channel, and two similar but different genres for the left and right channels (e.g. `Rock` and `Metal`).

              If you don’t want to use this feature, simply select the same genre for all three upsamplings.
            ''')

          with gr.Row():

            for input in [ UI.genre_for_upsampling_left_channel, UI.genre_for_upsampling_center_channel, UI.genre_for_upsampling_right_channel ]:

              input.render()
        
        UI.upsample_button.render().click(
          inputs = UI.upsampling_in_progress,
          outputs = [ UI.upsampling_in_progress, UI.upsampling_tracker_markdown, UI.upsample_button ],
          fn = lambda is_running: {
            UI.upsampling_in_progress: 0 if is_running else 1,
            UI.upsampling_tracker_markdown: 'Stopping upsampling...' if is_running else 'Upsampling...',
            UI.upsample_button: gr.update(
              value = 'Upsample' if is_running else 'Stop upsampling',
              variant = 'primary' if is_running else 'secondary',
            )
          },
          _js = """
            // Confirm before starting the upsample process
            (...args) => {
              console.log('Upsample button clicked with args: ', args)
              if ( !confirm('Are you sure you want to start the upsample process? THIS WILL TAKE A LONG TIME (see the warning above).') )
                throw new Error('Upsample process canceled by user')
              else {
                return args
              }
            }
          """
        )

        UI.upsampling_in_progress.render()
        
        UI.upsampling_in_progress.change(
          inputs = [
            UI.upsampling_in_progress, UI.project_name, UI.sample_to_upsample, UI.artist, UI.lyrics,
            UI.genre_for_upsampling_left_channel, UI.genre_for_upsampling_center_channel, UI.genre_for_upsampling_right_channel
          ],
          outputs = None,
          fn = toggle_upsampling,
          api_name = 'toggle-upsampling',
        )

        # upsampled_audio = gr.Audio(
        #   label = 'Upsampled sample (updated every few minutes)',
        #   visible = False,
        # )

        UI.upsampling_tracker_markdown.render()

        # Whenever the upsampling tracker changes (which is in turn triggered by an iteration of the upsample function), update the upsampled audio
        UI.upsampling_tracker_markdown.change(
          inputs = None,
          outputs = UI.upsampling_tracker_markdown,
          fn = lambda: f"{datetime.now().strftime('%H:%M:%S')}: Upsampling level {Upsampling.level}, window {Upsampling.window_index + 1}/{len(Upsampling.windows)}, at ~{Upsampling.time_per_window} s/window, ~{Upsampling.time_remaining // 60} minutes remaining" if Upsampling.time_remaining is not None else f"{datetime.now().strftime('%H:%M:%S')}: Estimating time remaining...",
          api_name = 'get-upsampled-audio',
          _js = """
            // Wait for 10 seconds before updating the tracker/upsampled audio
            async (...args) => {
              console.log('Waiting for 10 seconds before updating the tracker/upsampled audio')
              console.log('Args: ', args)
              await new Promise(resolve => setTimeout(resolve, 10000))
              console.log('Done waiting; updating.')
              // Also click the shadow root's internal-refresh-button
              // window.shadowRoot.getElementById('internal-refresh-button').click()
              return [ null ]
            }
          """
        )

      with gr.Tab('Panic'):

        with gr.Accordion('?', open = False):

          gr.Markdown('''
            Sometimes the app will crash due to insufficient GPU memory. If this happens, you can try using the button below to empty the cache.

            If that doesn’t work, you’ll have to restart the runtime (`Runtime` > `Restart and run all` in Colab). That’ll take a couple of minutes, but the memory will be new as a daisy.

            Usually around 12 GB of GPU RAM is needed to safely run the app.
          ''')

        memory_usage = gr.Textbox(
          label = 'GPU memory usage',
          value = 'Click Refresh to update',
        )

        def get_gpu_memory_usage():
          return subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader'],
            encoding='utf-8'
          ).strip()

        with gr.Row():
        
          gr.Button('Refresh').click(
            inputs = None,
            outputs = memory_usage,
            fn = get_gpu_memory_usage,
          )

          gr.Button('Empty cache', variant='primary').click(
            inputs = None,
            outputs = memory_usage,
            fn = lambda: [
              empty_cache(),
              get_gpu_memory_usage(),
            ][-1],
          )

        # Accordion for eval (danger zone)
        with gr.Accordion('Danger zone', open = False):

          gr.Markdown('''
            The following input box allows you to execute arbitrary Python code. IF YOU DON’T KNOW WHAT YOU’RE DOING, DON’T USE THIS FEATURE.
          ''')

          eval_code = gr.Textbox(
            label = 'Python code',
            placeholder = 'Shift+Enter for a new line',
            value = '',
            max_lines = 10,
          )

          eval_button = gr.Button('Execute')

          eval_output = gr.Textbox(
            label = 'Output',
            value = '',
            max_lines = 10,
          )

          eval_args = dict(
            inputs = eval_code,
            outputs = eval_output,
            fn = lambda code: eval(code),
          )

          eval_button.click(**eval_args)
          eval_code.submit(**eval_args)

  def dummy_string_input():
    return gr.Textbox( visible = False )
    
  app.load(
    on_load,
    inputs = [ gr.Textbox(visible=False), gr.Textbox(visible=False), gr.Textbox(visible=False) ],
    outputs = [ 
      UI.project_name, UI.routed_sample_id, UI.artist, UI.genre, UI.getting_started_column, UI.separate_tab_warning, UI.separate_tab_link, UI.main_window,
      UI.genre_for_upsampling_left_channel, UI.genre_for_upsampling_center_channel, UI.genre_for_upsampling_right_channel
    ],
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

        window.shadowRoot = document.querySelector('gradio-app').shadowRoot

        // The wavesurfer element is hidden inside a shadow DOM hosted by <gradio-app>, so we need to get it from there
        let shadowSelector = selector => window.shadowRoot.querySelector(selector)

        let waveformDiv = shadowSelector('#audio-waveform')
        console.log(`Found waveform div:`, waveformDiv)

        let timelineDiv = shadowSelector('#audio-timeline')
        console.log(`Found timeline div:`, timelineDiv)
        
        let getAudioTime = time => {
          let previewDuration = wavesurfer.getDuration()
          // console.log('Preview duration: ', previewDuration)
          // Take total duration from #total-audio-length's input, unless #trim-to-n-sec is set, in which case use that
          let trimToNSec = parseFloat(shadowSelector('#trim-to-n-sec input')?.value || 0)
          // console.log('Trim to n sec: ', trimToNSec)
          let totalDuration = trimToNSec || parseFloat(shadowSelector('#total-audio-length input').value)
          // console.log('Total duration: ', totalDuration)
          let additionalDuration = totalDuration - previewDuration
          // console.log('Additional duration: ', additionalDuration)
          let result = Math.round( ( time + additionalDuration ) * 100 ) / 100          
          // console.log('Result: ', result)
          return result
        }

        // Create a (global) wavesurfer object with and attach it to the div
        window.wavesurferTimeline = WaveSurfer.timeline.create({
          container: timelineDiv,
          // Light colors, as the background is dark
          primaryColor: '#eee',
          secondaryColor: '#ccc',
          primaryFontColor: '#eee',
          secondaryFontColor: '#ccc',
          formatTimeCallback: time => Math.round(getAudioTime(time))
        })

        window.wavesurfer = WaveSurfer.create({
          container: waveformDiv,
          waveColor: 'skyblue',
          progressColor: 'steelblue',
          plugins: [
            window.wavesurferTimeline
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

        // Put an observer on #audio-file (also in the shadow DOM) to reload the audio from its inner <a> element
        let parentElement = window.shadowRoot.querySelector('#audio-file')
        let parentObserver

        parentObserver = new MutationObserver( mutations => {
          // Check if there is an inner <a> element
          let wavesurferSrcElement = parentElement.querySelector('a')
          if ( wavesurferSrcElement ) {
            
            console.log('Found audio element:', wavesurferSrcElement)

            // If so, create an observer on it while removing the observer on the parent
            parentObserver.disconnect()

            let audioSrc

            let reloadAudio = () => {
              // Check if the audio source has changed
              if ( wavesurferSrcElement.href && wavesurferSrcElement.href !== audioSrc ) {
                // If so, reload the audio
                audioSrc = wavesurferSrcElement.href
                console.log('Reloading audio from', audioSrc)
                wavesurfer.load(audioSrc)
                // Also set the href of #download-button to the audio source
                window.shadowRoot.querySelector('#download-button').href = audioSrc
              }
            }

            reloadAudio()

            let audioObserver = new MutationObserver( reloadAudio )

            audioObserver.observe(wavesurferSrcElement, { attributes: true })

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