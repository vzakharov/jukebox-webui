import glob
import hashlib
import os
import random
import re
import shutil
import subprocess
import urllib.request
from datetime import datetime, timedelta, timezone

import gradio as gr
import librosa
import numpy as np
import torch as t
import yaml

from jukebox.hparams import Hyperparams
from jukebox.sample import load_prompts
from jukebox.utils.torch_utils import empty_cache
from lib.cut import cut_z, cut_zs, cut_audio
from lib.generate import generate

from lib.ui.UI import UI
from lib.upsampling.Upsampling import Upsampling
from lib.upsampling.start_upsampling import start_upsampling
from lib.load_model import load_model
from lib.upsampling.utils import get_first_upsampled_ancestor_zs
from lib.upsampling.utils import get_levels
from lib.utils import convert_name
from lib.upsampling.utils import is_upsampled
from params import DEV_MODE, GITHUB_SHA, base_path, debug_gradio, share_gradio, total_duration

raw_to_tokens = 128
chunk_size = 16
lower_batch_size = 16
lower_level_chunk_size = 32

hps = Hyperparams()
hps.sr = 44100
hps.levels = 3
hps.hop_fraction = [ 0.5, 0.5, 0.125 ]
hps.sample_length = int(total_duration * hps.sr // raw_to_tokens) * raw_to_tokens

### Model. Don't load if --no-load flag is set.
# Check if the flag is set when running the script (python app.py --no-load)
import sys

print("Launch arguments:", sys.argv)

if '--no-load' in sys.argv:
  print("🚫 Skipping model loading")
  pass

else:

  device, browser_timezone, keep_upsampling_after_restart, vqvae, priors, top_prior = load_model(hps)


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

def get_zs_filename(project_name, sample_name, with_prefix = True):
  if not with_prefix:
    sample_name = f'{project_name}-{sample_name}'
  return f'{base_path}/{project_name}/{sample_name}.z'

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
    UI.first_generation_row: HIDE,
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

def get_zs(project_name, sample_id, seek_upsampled = False):
  global base_path

  filename = f'{base_path}/{project_name}/{sample_id}.z'
  zs = t.load(filename)
  if not is_upsampled(zs) and seek_upsampled:
    upsampled_ancestor = get_first_upsampled_ancestor_zs(project_name, sample_id)
    if upsampled_ancestor:
      zs[:-1] = upsampled_ancestor[:-1]
  print(f'Loaded {filename}')
  return zs

def save_zs(zs, project_name, sample_id):
  global base_path

  filename = f'{base_path}/{project_name}/{sample_id}.z'
  t.save(zs, filename)
  print(f'Wrote {filename}')

def backup_sample(project_name, sample_id):
  global base_path

  current_filename = f'{base_path}/{project_name}/{sample_id}.z'
  timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
  backup_filename = f'{base_path}/{project_name}/bak/{sample_id}_{timestamp}.z'
  if not os.path.exists(os.path.dirname(backup_filename)):
    os.makedirs(os.path.dirname(backup_filename))
  # t.save(zs, backup_filename)
  shutil.copyfile(current_filename, backup_filename)
  print(f'Backed up {backup_filename}')

def get_audio(project_name, sample_id, cut_audio, preview_sec, level=None, stereo_rendering=3, combine_levels=True, invert_center=False):

  print(f'Generating audio for {project_name}/{sample_id} (level {level}, stereo rendering {stereo_rendering}, combine levels {combine_levels})')
  print(f'Cut: {cut_audio}, preview: {preview_sec}')

  # Get current GPU memory usage. If it's above 12GB, empty the cache
  memory = t.cuda.memory_allocated()
  print(f'GPU memory usage is {memory / 1e9:.1f} GB')
  if t.cuda.memory_allocated() > 12e9:
    print('GPU memory usage is above 12GB, clearing the cache')
    empty_cache()
    print(f'GPU memory usage is now {t.cuda.memory_allocated() / 1e9:1f} GB')

  global base_path, hps

  zs = get_zs(project_name, sample_id, seek_upsampled=True)

  # If no level is specified, use 2 (and then go downwards if combine_levels is True)
  if level is None:
    level = 2

  z = zs[level]
  # z is of shape torch.Size([1, n_tokens])
  # print(f'Loaded {filename}.z at level {level}, shape: {z.shape}')

  if cut_audio:
    z = cut_z(z, cut_audio, level)
  
  # Update audio_length
  audio_length = int( tokens_to_seconds(z.shape[1], level) * 100 ) / 100
  
  if preview_sec:
    seconds_to_cut_from_start = audio_length - abs(preview_sec) if preview_sec < 0 else preview_sec
    # For negative values, we need to replace "-" with "<" because "-" is used to indicate a range
    z = cut_z(z, f'-{seconds_to_cut_from_start}', level)
  else:
    seconds_to_cut_from_start = 0

  def decode(z):
    if z.shape[1] > 0:
      wav = vqvae.decode([ z ], start_level=level, end_level=level+1).cpu().numpy()
      # the decoded wav is of shape (n_samples, sample_length, 1). We will convert it later to (n_samples, 1 or 2 depending on stereo_rendering)
    else:
      # If the sample is empty, we need to create an empty wav of the right shape
      wav = np.zeros((z.shape[0], 0, 1))
    return wav
  
  # If z is longer than 30 seconds, there will likely be not enough RAM to decode it in one go
  # In this case, we'll split it into 30-second chunks (with a 5-second overlap), decode each chunk separately, and concatenate the results, crossfading the overlaps
  if z.shape[1] < seconds_to_tokens(30, level):
    wav = decode(z)
  else:
    chunk_size = seconds_to_tokens(30, level)
    overlap_size = seconds_to_tokens(5, level)
    print(f'z is too long ({z.shape[1]} tokens), splitting into chunks of {chunk_size} tokens, with a {overlap_size} token overlap')
    wav = None
    # Keep in mind that the last chunk can be shorter if the total length is not a multiple of chunk_size)
    for i in range(0, z.shape[1], chunk_size - overlap_size):

      # If this is the last chunk, make the chunk_size smaller if necessary
      overflow = i + chunk_size - z.shape[1]
      is_last_chunk = overflow > 0
      if is_last_chunk:
        chunk_size -= overflow
        # print(f'Last chunk, reduced chunk_size from {chunk_size + overflow} to {chunk_size} tokens')

      left_overlap_z = z[ :, i:i+overlap_size ]
      # print(f'Left overlap (tokens): {left_overlap_z.shape[1]}')
      left_overlap = decode(left_overlap_z)
      # print(f'Left overlap (quants): {left_overlap.shape[1]}')


      def fade(overlap, direction):
        # To fade in, we need to add 1/4 of the overlap as silence, 2/4 of the overlap as a linear ramp, and 1/4 of the overlap as full volume
        is_fade_in = direction == 'in'
        overlap_quants = overlap.shape[1]
        silence_quants = int( overlap_quants / 4 )
        ramp_quants = int( overlap_quants / 2 )
        if is_fade_in:
          overlap[:, :silence_quants, :] = 0
        else:
          overlap[:, -silence_quants:, :] = 0
        start = 0 if is_fade_in else 1
        overlap[:, silence_quants:-silence_quants, :] *= np.linspace(start, 1 - start, ramp_quants).reshape(1, -1, 1)
        return overlap

      if wav is not None:

        # Fade in the left overlap and add it to the existing wav if it's not empty (i.e. if this is not the first chunk)
        left_overlap = fade(left_overlap, 'in')
        # print(f'Faded in left overlap')
        # # Show as plot
        # plt.plot(left_overlap[0, :, 0])
        # plt.show()

        wav[ :, -left_overlap.shape[1]: ] += left_overlap
        # print(f'Added left overlap to existing wav:')
        # # Plot the resulting (faded-in + previous fade-out) overlap
        # plt.plot(wav[0, -left_overlap.shape[1]:, 0])
        # plt.show()

        print(f'Added left overlap to wav, overall shape now: {wav.shape}')

      else:
        wav = left_overlap
        print(f'Created wav with left overlap')

      # We'll also won't need right overlap for the last chunk
      main_chunk_z = z[ :, i+overlap_size: i+chunk_size-overlap_size if not is_last_chunk else i+chunk_size ]
      print(f'Main chunk (tokens): {main_chunk_z.shape[1]}')

      if main_chunk_z.shape[1] > 0:

        main_chunk = decode(main_chunk_z)
        print(f'Main chunk (quants): {main_chunk.shape[1]}')

        # Add the main chunk to the existing wav
        wav = np.concatenate([ wav, main_chunk ], axis=1)
        print(f'Added main chunk to wav, overall shape now: {wav.shape}')

      else:
        print('Main chunk is empty, skipping')
        continue

      # Fade out the right overlap, unless this is the last chunk
      if not is_last_chunk:

        right_overlap_z = z[ :, i+chunk_size-overlap_size:i+chunk_size ]
        # print(f'Right overlap (tokens): {right_overlap_z.shape[1]}')

        right_overlap = decode(right_overlap_z)
        # print(f'Right overlap (quants): {right_overlap.shape[1]}')

        right_overlap = fade(right_overlap, 'out')
        # print(f'Faded out right overlap')
        # # Show as plot
        # plt.plot(right_overlap[0, :, 0])
        # plt.show()

        # Add the right overlap to the existing wav
        wav = np.concatenate([ wav, right_overlap ], axis=1)
        # print(f'Added right overlap to wav, overall shape now: {wav.shape}')
      
      else:

        print(f'Last chunk, not adding right overlap')
        break
      
      print(f'Decoded {i+chunk_size} tokens out of {z.shape[1]}, wav shape: {wav.shape}')

  # wav is now of shape (n_samples, sample_length, 1)
  # If this is level 2, we want just (sample_length,), picking the first sample if there are multiple
  if level == 2:
    wav = wav[0, :, 0]

  # Otherwise, this is a batch of upsampled audio, so we need to act depending on the upsample_rendering parameter
  else:

    # upsample_rendering of 0, 1 or 2 means we just need to pick one of the samples
    if stereo_rendering < 3:

      wav = wav[stereo_rendering, :, 0]
    
    # upsample_rendering of 3 means we need to convert the audio to stereo, putting sample 0 to the left, 1 to the center, and 2 to the right
    # 4 means we also want to add a delay of 20 ms for the left and 40 ms for the right channel

    else:

      # def to_stereo(wav, stereo_delay_ms=0):
      def to_stereo(wav, stereo_delay_ms=0, invert_center=False):

        # A stereo wav is of form (sample_length + double the delay, 2)
        delay_quants = int( stereo_delay_ms * hps.sr / 1000 )
        stereo = np.zeros((wav.shape[1] + 2 * delay_quants, 2))
        # First let's convert the wav to [n_quants, n_samples] by getting rid of the last dimension and transposing the rest
        wav = wav[:, :, 0].T
        # print(f'Converted wav to shape {wav.shape}')
        # Take sample 0 for left channel (delayed once), 1 for both channels (non-delayed), and sample 2 for right channel (delayed twice)
        if delay_quants:
          stereo[ delay_quants: -delay_quants, 0 ] = wav[ :, 0 ]
          stereo[ 2 * delay_quants:, 1 ] = wav[ :, 2 ]
          stereo[ : -2 * delay_quants, 0 ] += wav[ :, 1 ]
          stereo[ : -2 * delay_quants, 1 ] += wav[ :, 1 ]
        else:
          stereo[ :, 0 ] = wav[ :, 0 ] + wav[ :, 1 ]
          stereo[ :, 1 ] = wav[ :, 2 ] + wav[ :, 1 ]
        
        if invert_center:
          stereo[ :, 1 ] *= -1

        # Now we have max amplitude of 2, so we need to divide by 2
        stereo /= 2

        # print(f'Converted to stereo with delay {stereo_delay_ms} ms, current shape: {stereo.shape}, max/min amplitudes: {np.max(stereo)}/{np.min(stereo)}')

        return stereo
      
      wav = to_stereo(wav, 20 if stereo_rendering == 4 else 0, invert_center)

  upsampled_lengths = [ 0, 0 ]
  if combine_levels:

    available_levels = get_levels(zs)
    combined_wav = None

    for sub_level in available_levels:

      if sub_level < level:
        sub_wav = get_audio(project_name, sample_id, cut_audio, seconds_to_cut_from_start, sub_level, stereo_rendering, combine_levels, invert_center)[0]
        upsampled_lengths[sub_level] = sub_wav.shape[0] / hps.sr + seconds_to_cut_from_start
      else:
        sub_wav = wav
        # If the wav is mono, we need to convert it to stereo by using the same values for both channels
        # (Note that this is most always the case, since the original audio is always mono, and this function is likely to be called for the original level, but we're abstracting it just in case)
        if sub_wav.ndim == 1:
          sub_wav = np.stack([ sub_wav, sub_wav ], axis=1)

      if combined_wav is None:
        combined_wav = sub_wav
        print(f'Created wav of length {combined_wav.shape[0]} for level {sub_level}')
      else:
        n_to_add = sub_wav.shape[0] - combined_wav.shape[0]
        # (This might be confusing why we are subtracting the shape of combined wav from the "sub" wav, but it's because the higher level "sub" wav is the one that is being upsampled, so it's the one that needs to be longer. The entire terminology with levels going backwards while the quality goes up is confusing, but we work with what we have)
        if n_to_add > 0:
          print(f'Adding {n_to_add} samples for level {sub_level}')
          combined_wav = np.concatenate([ combined_wav, sub_wav[ -n_to_add: ] ], axis=0)

    wav = combined_wav

  print(f'Generated audio of length {len(wav)} ({ len(wav) / hps.sr } seconds); original length: {audio_length} seconds.')

  return wav, audio_length, upsampled_lengths

# def get_audio_being_upsampled():

#   if not Upsampling.running:
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
    match = re.match(f'{prefix}(\d+)\\.zs?$', filename)
    if match:
      child_ids += [ filename.split('.')[0] ]
    
  if include_custom:

    custom_parents = get_custom_parents(project_name)

    for sample_id in custom_parents:
      if custom_parents[sample_id] == parent_sample_id:
        child_ids += [ sample_id ]        

  # print(f'Children of {parent_sample_id}: {child_ids}')

  # Sort alphabetically
  child_ids.sort()

  return child_ids

def get_custom_parents(project_name, force_reload=False):

  global base_path, custom_parents
  
  if not custom_parents or custom_parents['project_name'] != project_name or force_reload:
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
    UI.genre_dropdown: gr.update(
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

def get_samples(project_name, leafs_only):

  choices = []
  for filename in os.listdir(f'{base_path}/{project_name}'):
    if re.match(r'.*\.zs?$', filename):
      id = filename.split('.')[0]
      if leafs_only and len( get_children(project_name, id) ) > 0:
        continue
      choices += [ id ]
  
  # Sort by id, in descending order
  choices.sort(reverse = True)
  
  return choices

def get_projects(include_new = True):
  
  global base_path

  # print(f'Getting project list for {base_path}...')

  project_names = []
  for folder in os.listdir(base_path):
    if os.path.isdir(base_path+'/'+folder) and not folder.startswith('_'):
      project_names.append(folder)
  # Sort project names alphabetically
  project_names.sort()

  print(f'Found {len(project_names)} projects: {project_names}')

  if include_new:
    project_names = ['CREATE NEW', *project_names]

  return project_names

def get_project_name_from_sample_id(sample_id):
  projects = get_projects(include_new = False)
  # Find a project that matches the sample id, which is [project name]-[rest of sample id]
  for project_name in projects:
    if sample_id.startswith(f'{project_name}-'):
      return project_name

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

    project_path = f'{base_path}/{project_name}'
    hps.name = project_path
    settings_path = f'{project_path}/{project_name}.yaml'
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

def get_sample_filename(project_name, sample_id, cut_out, last_n_sec, upsample_rendering, combine_levels, invert_center):
    
    filename = f'{base_path}/{project_name}/rendered/{sample_id} '

    # Add cutout/preview suffixes, replacing dots with underscores (to avoid confusion with file extensions)

    def replace_dots_with_underscores(number):
      return str(number).replace('.', '_')

    if cut_out:
      filename += f'cut {replace_dots_with_underscores(cut_out)} '
    if last_n_sec:
      filename += f'last {replace_dots_with_underscores(last_n_sec)} '
    
    # Add lowercase of upsample rendering option
    if upsample_rendering:
      filename += f'r{upsample_rendering} '
    
    if combine_levels:
      filename += f'combined '

    if invert_center:
      filename += 'inverted '
    
    return filename

def get_sample(project_name, sample_id, cut_out='', last_n_sec=None, upsample_rendering=4, combine_levels=True, invert_center=False, force_reload=False):

  global hps

  print(f'Loading sample {sample_id}')

  filename = get_sample_filename(project_name, sample_id, cut_out, last_n_sec, upsample_rendering, combine_levels, invert_center)
  filename_without_hash = filename

  # Add a hash (8 characters of md5) of the corresponding z file (so that we can detect if the z file has changed and hence we need to re-render)
  filename += f'{hashlib.md5(open(f"{base_path}/{project_name}/{sample_id}.z", "rb").read()).hexdigest()[:8]} '


  print(f'Checking if {filename} is cached...')

  # Make sure all 3 of wav, mp3 and yaml exist
  if not force_reload:
    for ext in [ 'wav', 'mp3', 'yaml' ]:
      if not os.path.isfile(f'{filename}.{ext}'):
        force_reload = True
        break

  if force_reload:

    print('Nope, rendering...')

    # First, let's delete any old files that have the same name but a different hash (because these are now obsolete)
    for f in glob.glob(f'{filename_without_hash}.*'):
      print(f'(Deleting now-obsolete cached file {f})')
      os.remove(f)

    if last_n_sec:
      last_n_sec = -last_n_sec
    # (Because get_audio, called below, accepts "preview_sec", whose positive value means "preview the first n seconds", but we want to preview the last n seconds)

    wav, total_audio_length, upsampled_lengths = get_audio(project_name, sample_id, cut_out, last_n_sec, None, upsample_rendering, combine_levels, invert_center)
    # (level is None, which means "the highest level that is available", i.e. 2)

    if not os.path.exists(os.path.dirname(filename)):
      os.makedirs(os.path.dirname(filename))

    librosa.output.write_wav(f'{filename}.wav', np.asfortranarray(wav), hps.sr)

    # Add metadata (total audio length, upsampled lengths) to a yaml with the same name as the wav file
    with open(f'{filename}.yaml', 'w') as f:
      yaml.dump({
        'total_audio_length': total_audio_length,
        'upsampled_lengths': upsampled_lengths
      }, f)

    # Convert to mp3
    subprocess.run(['ffmpeg', '-y', '-i', f'{filename}.wav', '-acodec', 'libmp3lame', '-ab', '320k', f'{filename}.mp3'])

    # If there are more than 30 files in the rendered folder (i.e. more than 10 samples), delete the ones older than 1 day
    file_count_limit = 30
    files = glob.glob(f'{base_path}/{project_name}/rendered/*')
    if len(files) > file_count_limit:
      removed_count = 0
      failed_count = 0
      for f in files:
        try:
          if datetime.now() - datetime.fromtimestamp(os.path.getmtime(f)) > timedelta(days=1):
            os.remove(f)
            removed_count += 1
        except Exception as e:
          print(f'Could not remove {f}: {e}')
          failed_count += 1
      print(f'Deleted {removed_count} of {removed_count + failed_count} old files')
        
  else:
    print('Yep, using it.')
    wav = None

    # Load metadata
    with open(f'{filename}.yaml', 'r') as f:
      metadata = yaml.load(f, Loader=yaml.FullLoader)
      total_audio_length = metadata['total_audio_length']
      upsampled_lengths = metadata['upsampled_lengths']
      print(f'(Also loaded metadata: {metadata})')

  chunk_filenames = []

  # If the mp3 size is > certain sie, we'll need to send it back in chunks, so we divide the mp3 into as many chunks as needed
  file_size = os.path.getsize(f'{filename}.mp3')
  file_limit = 300000

  if file_size > file_limit:
    print(f'MP3 file size is {file_size} bytes, splitting into chunks...')
    file_content = open(f'{filename}.mp3', 'rb').read()
    for i in range(0, file_size, file_limit):
      # Place the chunk file in tmp/[filename without path] [range].mp3_chunk
      # Create the tmp folder if it doesn't exist
      if not os.path.exists(f'tmp'):        
        os.makedirs(f'tmp')
      chunk_filename = f'tmp/{os.path.basename(filename)}{i}-{i+file_limit} .mp3_chunk'
      with open(chunk_filename, 'wb') as f:
        f.write(file_content[i:i+file_limit])
        print(f'Wrote bytes {i}-{i+file_limit} to {chunk_filename}')
      chunk_filenames.append(chunk_filename)
  else:
    chunk_filenames = [f'{filename}.mp3']
  
  print(f'Files to send: {chunk_filenames}')

  return {
    UI.current_chunks: chunk_filenames,
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
    UI.upsampled_lengths: ','.join([str(length) for length in upsampled_lengths]),
    # Random number for picked sample updated flag
    UI.picked_sample_updated: random.random(),
  }

def get_sibling_samples(project_name, sample_id, cut_out, last_n_sec, upsample_rendering, combine_levels, invert_center):
  print(f'Updating sibling samples for {sample_id}...')
  sibling_files = []
  for sibling_id in get_siblings(project_name, sample_id):
    if sibling_id == sample_id:
      continue
    sibling_sample = get_sample(project_name, sibling_id, cut_out, last_n_sec, upsample_rendering, combine_levels, invert_center)
    sibling_sample_files = sibling_sample[UI.current_chunks]
    # breakpoint()
    print(f'Adding sibling {sibling_id} with files {sibling_sample_files}')
    sibling_files.extend(sibling_sample_files)
    print(f'Totally {len(sibling_files)} sibling files')
  return {
    UI.sibling_chunks: sibling_files
  }

def refresh_siblings(project_name, sample_id):
  
  if not sample_id:
    return {
      UI.picked_sample: HIDE
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

  # If the file already exists, raise an error
  if os.path.isfile(f'{base_path}/{project_name}/{new_sample_id}.z'):
    raise ValueError(f'{new_sample_id} already exists')

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

def is_ancestor(project_name, potential_ancestor, potential_descendant):
  parent = get_parent(project_name, potential_descendant)
  if parent == potential_ancestor:
    return True
  elif parent:
    return is_ancestor(project_name, potential_ancestor, parent)
  else:
    return False

def set_keep_upsampling_after_restart():
  global keep_upsampling_after_restart

  keep_upsampling_after_restart = True

def tokens_to_seconds(tokens, level = 2):

  global hps, raw_to_tokens

  return tokens * raw_to_tokens / hps.sr / 4 ** (2 - level)

SHOW = gr.update( visible = True )
HIDE = gr.update( visible = False )
SHOW_OR_HIDE = lambda x: gr.update( visible = x )

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
    }

    #audio-timeline {
      /* hide for now */
      display: none;
    }


  """,
  title = f'Jukebox Web UI { GITHUB_SHA }{ " (dev mode)" if DEV_MODE else "" }',
) as app:

  UI.browser_timezone.render()

  # Render an invisible checkbox group to enable loading list of projects via API
  project_list = gr.CheckboxGroup(
    visible = False,
  )

  project_list.change(
    inputs = None,
    outputs = [ project_list ],
    fn = lambda: get_projects(),
    api_name = 'get-projects'
  )

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
          
          elif component == UI.genre:

            UI.genre_dropdown.render().change(
              inputs = [ UI.genre, UI.genre_dropdown ],
              outputs = UI.genre,
              # Add after a space, if not empty
              fn = lambda genre, genre_dropdown: ( genre + ' ' if genre else '' ) + genre_dropdown,
            )
          
            component.render()

          elif component == UI.generation_discard_window:

            component.render()

            with gr.Accordion( 'What is this?', open = False ):

              gr.Markdown("""
                If your song is too long, the generation may take too much memory and crash. In this case, you can discard the first N seconds of the song for generation purposes (i.e. the model won’t take them into account when generating the rest of the song).
                          
                If your song has lyrics, put '---' (with a new line before and after) at the point that is now the “beginning” of the song, so that the model doesn’t get confused by the now-irrelevant lyrics.
              """)

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

      with gr.Tab('Workspace'):

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
                  UI.first_generation_row: HIDE,
                  UI.sample_tree_row: SHOW,
                }
              )

          with UI.sample_tree_row.render():
            
            UI.routed_sample_id.render()
            UI.sample_tree.render()

            with gr.Column():

              # with gr.Accordion('Options & stats', open=False ):

              UI.show_leafs_only.render()

              UI.show_leafs_only.change(
                inputs = [ UI.project_name, UI.show_leafs_only ],
                outputs = UI.sample_tree,
                fn = lambda *args: gr.update( choices = get_samples(*args) ),
              )

                # UI.branch_sample_count.render()
                # UI.leaf_sample_count.render()

                # # Recount on sample_tree change
                # UI.sample_tree.change(
                #   inputs = UI.project_name,
                #   outputs = [ UI.branch_sample_count, UI.leaf_sample_count ],
                #   fn = lambda project_name: [
                #     len(get_samples(project_name, leafs_only)) for leafs_only in [ False, True ]
                #   ]
                # )
            
          UI.picked_sample.render()

          UI.sample_tree.change(
            inputs = [ UI.project_name, UI.sample_tree ],
            outputs = UI.picked_sample,
            fn = refresh_siblings,
            api_name = 'get-siblings'        
          )

          preview_inputs = [
              UI.project_name, UI.picked_sample, UI.cut_audio_specs, UI.preview_just_the_last_n_sec,
              UI.upsample_rendering, UI.combine_upsampling_levels, UI.invert_center_channel
          ]

          get_preview_args = lambda force_reload: dict(
            inputs = [
              *preview_inputs, gr.State(force_reload)
            ],
            outputs = [
              UI.sample_box, UI.current_chunks, #UI.generated_audio,
              UI.total_audio_length, UI.upsampled_lengths,
              UI.go_to_children_button, UI.go_to_parent_button,
              UI.picked_sample_updated
            ],
            fn = get_sample,
          )

          default_preview_args = get_preview_args(False)

          # Virtual input & handler to create an API method for get_sample_filename
          gr.Textbox(visible=False).change(
            inputs = preview_inputs,
            outputs = gr.Textbox(visible=False),
            fn = get_sample_filename,
            api_name = 'get-sample-filename'
          )

          UI.picked_sample.change(
            **default_preview_args,
            api_name = 'get-sample',
            _js =
            # Set the search string to ?[sample_id] for easier navigation
            '''
              async ( ...args ) => {

                try {

                  let sample_id = args[1]

                  sample_id && window.history.pushState( {}, '', `?${args[1]}` )

                  // Gray out the wavesurfer
                  Ji.grayOutWavesurfer()

                  // Now we'll try to reload the audio from cache. To do that, we'll find the first cached blob (Ji.blobCache) whose key starts with the sample_id either followed by space or end of string.
                  // (Although different version of the same sample might have been cached, the first one will be the one that was added last, so it's the most recent one)
                  let cached_blob = Ji.blobCache.find( ({ key }) => key.match( new RegExp(`^${sample_id}( |$)`) ) )
                  if ( cached_blob ) {
                    console.log( 'Found cached blob', cached_blob )
                    let { key, blob } = cached_blob
                    wavesurfer.loadBlob( blob )
                    Ji.lastLoadedBlobKey = key
                    Ji.preloadedAudio = true
                    // Gray out slightly less
                    Ji.grayOutWavesurfer( true, 0.75 )
                  }

                } catch (e) {
                  console.error(e)
                } finally {

                  return args

                }


              }
            '''
          )
           
          # When the picked sample is updated, update all the others too (UI.sibling_chunks) by calling get_sample for each sibling
          UI.picked_sample_updated.render().change(
            inputs = [ *preview_inputs ],
            outputs = UI.sibling_chunks,
            fn = get_sibling_samples,
            api_name = 'get-sibling-samples',
          )

          UI.current_chunks.render()
          UI.sibling_chunks.render()

          UI.upsampled_lengths.render().change(
            inputs = UI.upsampled_lengths,
            outputs = None,
            fn = None,
            # Split by comma and turn into floats and add wavesurfer markers for each (first clear all the markers)
            _js = 'comma_separated => Ji.addUpsamplingMarkers( comma_separated.split(",").map( parseFloat ) )'
          )

          with UI.sample_box.render():

            with UI.upsampling_accordion.render():

              with gr.Row():

                with gr.Column():

                  UI.upsampling_level.render().change(
                    **default_preview_args,
                  )

                  # Only show the upsampling elements if there are upsampled versions of the picked sample
                  def show_or_hide_upsampling_elements(project_name, sample_id, upsampling_running):

                    levels = get_levels(get_zs(project_name, sample_id))
                    # print(f'Levels: {levels}')

                    available_level_names = UI.UPSAMPLING_LEVEL_NAMES[:len(levels)]
                    print(f'Available level names: {available_level_names}')

                    return {
                      # UI.upsampling_accordion: gr.update(
                      #   visible = len(levels) > 1 or upsampling_running,
                      # ),
                      # (removing the accordion for now)
                      UI.upsampling_status: gr.update(
                        visible = upsampling_running,
                      ),
                      UI.upsampling_level: gr.update(
                        choices = available_level_names,
                        # Choose the highest available level
                        value = available_level_names[-1],
                      )
                    }
                  
                  show_or_hide_upsampling_elements_args = dict(
                    inputs = [ UI.project_name, UI.picked_sample, UI.upsampling_running ],
                    outputs = [ UI.upsampling_status, UI.upsampling_level ],
                    fn = show_or_hide_upsampling_elements,
                  )

                  UI.picked_sample.change( **show_or_hide_upsampling_elements_args )
                  UI.upsampling_running.change( **show_or_hide_upsampling_elements_args )

                with gr.Column() as upsampling_manipulation_column:

                  # # Show the column only if an upsampled sample is selected and hide the compose row respectively (we can only compose with the original sample)
                  # UI.upsampling_level.change(
                  #   inputs = [ UI.upsampling_level, UI.upsampling_running ],
                  #   outputs = [ upsampling_manipulation_column, UI.compose_row ],
                  #   fn = lambda upsampling_level, upsampling_running: [
                  #     gr.update( visible = upsampling_level != 'Raw' ),
                  #     gr.update( visible = upsampling_level == 'Raw' and not upsampling_running ),
                  #   ]
                  # )

                  with gr.Row():

                    UI.upsample_rendering.render().change(
                      **default_preview_args,
                    )

                    UI.combine_upsampling_levels.render().change(
                      **default_preview_args,
                    )

                    UI.invert_center_channel.render().change(
                      **default_preview_args,
                    )

              # Show the continue upsampling markdown only if the current level's length in tokens is less than the total audio length
              # Also update the upsampling button to say "Continue upsampling" instead of "Upsample"
              def show_or_hide_continue_upsampling(project_name, sample_id, total_audio_length, upsampling_running):

                if not upsampling_running:
                  zs = get_zs(project_name, sample_id)
                  levels = get_levels(zs)
                  # print(f'Levels: {levels}, z: {zs}')
                  # We'll show if there's no level 0 in levels or if the length of level 0 (in seconds) is less than the length of level 2 (in seconds)
                  must_show = 0 not in levels or tokens_to_seconds(len(zs[0]), 0) < tokens_to_seconds(len(zs[2]), 2)
                  # print(f'Must show: {must_show}')
                  
                else:
                  must_show = True

                return gr.update( visible = must_show )
              
              UI.picked_sample.change(
                inputs = [ UI.project_name, UI.picked_sample, UI.total_audio_length, UI.upsampling_running ],
                outputs = UI.continue_upsampling_button,
                fn = show_or_hide_continue_upsampling,
              )

              upsample_button_click_args = dict(
                inputs = UI.upsampling_running,
                outputs = [ UI.upsampling_running, UI.upsampling_triggered_by_button ],
                fn = lambda was_running: 
                # If was running (i.e. we're stopping), kill the runtime (after a warning) and show an alert saying to restart the runtime in Colab
                  [
                    print('Killing runtime...'),
                    subprocess.run(['kill', '-9', str(os.getpid())]),
                  ] if was_running else {
                    UI.upsampling_running: 1,
                    UI.upsampling_triggered_by_button: True,
                  },
                _js = """
                  // Confirm before starting/stopping the upsample process
                  running => {
                    confirmText = 
                      running ?
                        'Are you sure you want to stop the upsample process? ⚠️ THIS WILL KILL THE RUNTIME AND YOU WILL HAVE TO RESTART IT IN COLAB ⚠️ (But your current upsampling progress will be saved)' :
                        'Are you sure you want to start the upsample process? THIS WILL TAKE HOURS, AND YOU WON’T BE ABLE TO CONTINUE COMPOSING!'
                    if ( !confirm(confirmText) ) {
                      throw new Error(`${running ? 'Stopping' : 'Starting'} upsample process canceled by user`)
                    } else {
                      // If running, show a message saying to restart the runtime in Colab
                      if ( running ) {
                        alert('Upsample process stopped. Please re-run the cell in Colab to restart the UI')
                      }
                      return [ running ]
                    }
                  }
                """
              )

              UI.continue_upsampling_button.render().click( **upsample_button_click_args )

              UI.upsampling_audio_refresher.render()

              def reset_audio_refresher():
                Upsampling.should_refresh_audio = False

              [ 
                UI.upsampling_audio_refresher.change( **action ) for action in [ 
                  default_preview_args, 
                  show_or_hide_upsampling_elements_args,
                  dict(
                    inputs = None,
                    outputs = None,
                    # Reset Upsampling.should_refresh_audio to False
                    fn = reset_audio_refresher
                  )
                ] 
              ]

            UI.upsampling_status.render()
            
            # Refresh button
            internal_refresh_button = gr.Button('🔃', elem_id = 'internal-refresh-button', visible=False)
            
            internal_refresh_button.click(
              **get_preview_args(force_reload = True),
            )

            internal_refresh_button.click(
              **show_or_hide_upsampling_elements_args,
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
              <input type="number" class="gr-box gr-input gr-text-input" id="audio-time" value="0">

              <!-- Download button -- it will be set to the right href later on -->
              <!--
              <a class="gr-button gr-button-lg gr-button-secondary" id="download-button">
                🔗
              </a>
              -->
              <!-- (Removed for now, as it only links to the first chunk, will fix later) -->

              <!-- Refresh button -- it virtually clicks the "internal-refresh-button" button (which is hidden) -->
              <button class="gr-button gr-button-lg gr-button-secondary" onclick="window.shadowRoot.getElementById('internal-refresh-button').click()" id="refresh-button">
                ↻
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

            with gr.Accordion( 'Advanced', open = False ):

              with gr.Tab('Manipulate audio'):

                UI.total_audio_length.render()

                # Change the max n samples depending on the audio length
                def set_max_n_samples( total_audio_length, n_samples ):

                  max_n_samples_by_gpu_and_duration = {
                    'Tesla T4': {
                      0: 4,
                      8.5: 3,
                      13: 2
                    }
                    # The key indicates the audio length threshold in seconds trespassing which makes max_n_samples equal to the value
                  }

                  # Get GPU via nvidia-smi
                  gpu = subprocess.check_output( 'nvidia-smi --query-gpu=gpu_name --format=csv,noheader', shell=True ).decode('utf-8').strip()

                  # The default value is 4
                  max_n_samples = 4
                  if gpu in max_n_samples_by_gpu_and_duration and total_audio_length:
                    # Get the max n samples for the GPU from the respective dict
                    max_n_samples_for_gpu = max_n_samples_by_gpu_and_duration[gpu]
                    max_n_samples_above_threshold = [ max_n_samples_for_gpu[threshold] for threshold in max_n_samples_for_gpu if total_audio_length > threshold ]
                    if len(max_n_samples_above_threshold) > 0:
                      max_n_samples = min( max_n_samples_for_gpu[ threshold ] for threshold in max_n_samples_for_gpu if total_audio_length > threshold )

                  return max_n_samples

                # Do this on audio length change and app load
                for handler in [ UI.total_audio_length.change, app.load ]:
                  handler(
                    inputs = [ UI.total_audio_length, UI.n_samples ],
                    outputs = UI.max_n_samples,
                    fn = set_max_n_samples,
                  )
                
                # If max_n_samples changes, update the n_samples input respectively
                UI.max_n_samples.render().change(
                  inputs = UI.max_n_samples,
                  outputs = UI.n_samples,
                  fn = lambda max_n_samples: gr.update(
                    maximum = max_n_samples,
                    # value = min( n_samples, max_n_samples ),
                  )
                )

                UI.cut_audio_specs.render().submit(**default_preview_args)

                with gr.Row():

                  UI.cut_audio_preview_button.render().click(**default_preview_args)

                  # Make the cut out buttons visible or not depending on whether the cut out value is 0
                  UI.cut_audio_specs.change(
                    inputs = UI.cut_audio_specs,
                    outputs = [ UI.cut_audio_preview_button, UI.cut_audio_apply_button ],
                    fn = lambda cut_audio_specs: [
                      gr.update( visible = cut_audio_specs != '' ) for _ in range(3)
                    ]
                  )

                  UI.cut_audio_apply_button.render().click(
                    inputs = [ UI.project_name, UI.picked_sample, UI.cut_audio_specs ],
                    outputs = UI.cut_audio_specs,
                    fn = cut_audio,
                    api_name = 'cut-audio',
                  )

                with gr.Accordion('How does it work?', open = False):
                  # possible specs:
                  # start-end -- cuts out the specified range (in seconds), either can be omitted to cut from the start or to the end, the dash can be omitted to cut from the specified time to the end
                  # start-end+sample_id@start-end -- takes the specified range from the specified sample and adds it instead of the cut-out range
                  # start-end+start-end -- same, but takes the specified range from the current sample
                  # +sample_id@start-end -- adds the specified range from the specified sample to the end of the current sample
                  # +start-end -- keeps just the specified range from the current sample (i.e. the opposite of start-end)
                  # Any whitespaces are ignored
                  gr.Markdown('''
                    - `start-end` (e.g. 0.5-2.5) — *removes* the specified range (in seconds),
                      - `start-` or just `start` — *removes* from the specified time to the end
                      - `-end` -- **removes** from the start to the specified time
                    - `start-end+start-end` — *removes* the range before `+` and *inserts* the range after `+` instead. Note that, unlike the remove range, the insert range must be fully specified.
                    - `start-end+sample_id@start-end` — same as above, but the insert range is taken from the specified sample, even if it is in another project (mix and match!)
                    - `+sample_id@start-end` — same as above, but the range from the other sample is added *to the end* of the current sample
                    - `+start-end` — *keeps* just the specified range and removes everything else.

                    You can combine several of the above by using a comma (`,`). **KEEP IN MIND** that in this case the ranges are applied sequentially, so the order matters. For example, `0-1,2-3` will first remove 0-1s, and will then remove 2-3s FROM THE ALREADY MODIFIED SAMPLE, so it will actually remove ranges 0-1s and *3-4s* from the original sample. This is intentional, as it allows for a step-by-step approach to editing the audio, where you add new specifiers as you listen to the result of the previous ones.
                  ''')

                UI.preview_just_the_last_n_sec.render().blur(**default_preview_args)

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
        
              with gr.Tab('Purge samples'):

                # For all samples whose parent sample's level 0/1 are the same as this one, purge those levels

                # We need a button to prepare the list of samples to purge, a multiline textbox to show the list, and a button to confirm the purge

                purge_list = gr.Textbox(
                  label = 'Purge list',
                  placeholder = 'Click the button below to prepare the list of samples to purge',
                  multiline = True,
                  disabled = True,
                )

                def prepare_purge_list(project_name):
                  # Get the list of samples to purge
                  # For each sample, get its parent sample
                  # If the parent sample's level 0/1 is the same as this one, add it to the list
                  samples = get_samples(project_name, False)
                  purge_list = []
                  for sample in samples:
                    try:
                      current_zs = t.load(get_zs_filename(project_name, sample))
                      parent = get_parent(project_name, sample)
                      if not parent:
                        print(f'No parent for {sample}, skipping')
                        continue
                      parent_zs = t.load(get_zs_filename(project_name, parent))
                      if ( 
                        t.equal( current_zs[0], parent_zs[0] ) and
                        t.equal( current_zs[1], parent_zs[1] ) and
                        ( current_zs[0].shape[1] + current_zs[1].shape[1] > 0 )
                      ):
                        purge_list.append(sample)
                        print(f'Adding {sample} to the purge list')
                    except Exception as e:
                      print(f'Error while processing {sample}: {e}, continuing')
                      continue
                  print(f'Purge list: {purge_list}')
                  return '\n'.join(purge_list)

                gr.Button('Prepare purge list').click(
                  inputs = [ UI.project_name ],
                  outputs = purge_list,
                  fn = prepare_purge_list,
                  api_name = 'prepare-purge-list'
                )

                def purge_samples(project_name, purge_list):
                  # Check the project size before purging, i.e. the joint size of all *.z files in the project directory
                  project_size_before = subprocess.check_output(['du','-sh', os.path.join( base_path, project_name )]).split()[0].decode('utf-8')
                  # For each sample in the purge list, delete it
                  for sample in purge_list.split('\n'):
                    zs = t.load(get_zs_filename(project_name, sample))
                    level2_shape0 = zs[2].shape[0]
                    purged_zs = [
                      t.empty([level2_shape0,0], device='cuda'),
                      t.empty([level2_shape0,0], device='cuda'),
                      zs[2]
                    ]
                    print(f'Purged {sample} from shapes {[x.shape for x in zs]} to {[x.shape for x in purged_zs]}')
                    t.save(purged_zs, get_zs_filename(project_name, sample))
                  # Check the project size after purging
                  project_size_after = subprocess.check_output(['du','-sh', os.path.join( base_path, project_name )]).split()[0].decode('utf-8')
                  return f'Project size before: {project_size_before}, after: {project_size_after}'
                
                gr.Button('Purge samples').click(
                  inputs = [ UI.project_name, purge_list ],
                  outputs = purge_list,
                  fn = purge_samples,
                  api_name = 'purge-samples'
                )

              with gr.Tab('Completify sample'):

                gr.Markdown('''
                  For space saving purposes, the app will sometime NOT include the entire information needed to render the sample into the sample file, taking the missing info (e.g. upsampled tokens) from its ancestors instead.
                            
                  If, for whatever reason, you want to have the entire information in the sample file, you can add it by clicking the button below.
                ''')

                def completify(project_name, sample_id):
                  zs = get_zs(project_name, sample_id, True)
                  backup_sample(project_name, sample_id)
                  save_zs(zs, project_name, sample_id)

                completify_button = gr.Button('Completify')
                completify_button.click(
                  completify,
                  [ UI.project_name, UI.picked_sample ],
                  gr.Button('Completify')
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
          outputs = [ UI.sample_tree, prime_button, UI.prime_timestamp, UI.first_generation_row ], # UI.prime_timestamp is updated to the current time to force tab change
          fn = convert_audio_to_sample,
          api_name = 'convert-wav-to-sample'
        )

        UI.prime_timestamp.render().change(
          inputs = UI.prime_timestamp, outputs = None, fn = None,
          _js = 
            # Find a button inside a div inside another div with class 'tabs', the button having 'Workspace' as text, and click it -- all this in the shadow DOM.
            # Gosh, this is ugly.
            """
              timestamp => {
                console.log(`Timestamp changed to ${timestamp}; clicking the 'Workspace' tab`)
                Ji.clickTabWithText('Workspace')
                return timestamp
              }
            """
        )

      with gr.Tab('Upsample'):

        # Warning that this process is slow and can take up to 10 minutes for 1 second of audio
        with gr.Accordion('What is this?', open = False):

          gr.Markdown('''
            Upsampling is a process that creates higher-quality audio from your composition. It is done in two steps:

            - “Midsampling,” which considerably improves the quality of the audio, takes around 2 minutes per one second of audio.

            - “Upsampling,” which improves the quality some more, goes after midsampling and takes around 8 minutes per one second of audio.

            Thus, say, for a one-minute song, you will need to wait around 2 hours to have the midsampled version, and around 8 hours _more_ to have the upsampled version.

            You will be able to listen to the audio as it is being generated: Each “window” of upsampling takes ~6 minutes and will give you respectively ~2.7 and ~0.7 additional seconds of mid- or upsampled audio to listen to.

            ⚠️ WARNING: As the upsampling process uses a different model, which cannot be loaded together with the composition model due to memory constraints, **you will not be able to upsample and compose at the same time**. To go back to composing you will have to restart the Colab runtime or start a second Colab runtime and use them in parallel.
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
          
          UI.kill_runtime_once_done.render()
        
        # If upsampling is running, enable the upsampling_refresher -- a "virtual" input that, when changed, will update the upsampling_status_markdown
        # It will do so after waiting for 10 seconds (using js). After finishing, it will update itself again, causing the process to repeat.
        UI.upsampling_refresher.render().change(
          inputs = [ UI.upsampling_refresher, UI.upsampling_audio_refresher ],
          outputs = [ UI.upsampling_refresher, UI.upsampling_status, UI.upsampling_audio_refresher ],
          fn = lambda refresher, audio_refresher: {
            UI.upsampling_status: Upsampling.status_markdown,
            UI.upsampling_refresher: refresher + 1,
            UI.upsampling_audio_refresher: audio_refresher + 1 if Upsampling.should_refresh_audio else audio_refresher
          },
          _js = """
            async ( ...args ) => {
              await new Promise( resolve => setTimeout( resolve, 10000 ) )
              console.log( 'Checking upsampling status...' )
              return args
            }
          """
        )

        UI.upsample_button.render().click( **upsample_button_click_args )

        # During app load, set upsampling_running and upsampling_stopping according to Upsampling.running
        app.load(
          inputs = None,
          outputs = UI.upsampling_running,
          fn = lambda: Upsampling.running,
        )
        
        UI.upsampling_triggered_by_button.render()

        # When upsampling_running changes via the button, run the upsampling process
        UI.upsampling_running.render().change(
          inputs = [
            UI.upsampling_triggered_by_button,
            UI.project_name, UI.sample_to_upsample, UI.artist, UI.lyrics,
            UI.genre_for_upsampling_left_channel, UI.genre_for_upsampling_center_channel, UI.genre_for_upsampling_right_channel,
            UI.kill_runtime_once_done            
          ],
          outputs = None,
          fn = lambda triggered_by_button, *args: start_upsampling( *args ) if triggered_by_button else None,
          api_name = 'toggle-upsampling',
          # Also go to the "Workspace" tab (because that's where we'll display the upsampling status) via the Ji.clickTabWithText helper method in js
          _js = """
            async ( ...args ) => {
              console.log( 'Upsampling toggled, args:', args )
              if ( args[0] ) {
                Ji.clickTabWithText( 'Workspace' )
                return args
              } else {
                throw new Error('Upsampling not triggered by button')
              }
            }
          """
        )

        # When it changes regardless of the session, e.g. also at page refresh, update the various relevant UI elements, start the refresher, etc.
        UI.upsampling_running.change(
          inputs = None,
          outputs = [ UI.upsampling_status, UI.upsample_button, UI.continue_upsampling_button, UI.upsampling_refresher ],
          fn = lambda: {
            UI.upsampling_status: 'Upsampling in progress...',
            UI.upsample_button: gr.update(
              value = 'Stop upsampling',
              variant = 'secondary',
            ),
            UI.continue_upsampling_button: gr.update(
              value = 'Stop upsampling',
            ),
            # Random refresher value (int) to trigger the refresher
            UI.upsampling_refresher: random.randint( 0, 1000000 ),
            # # Hide the compose row
            # UI.compose_row: HIDE,
          }
        )

      with gr.Tab('Panic'):

        with gr.Accordion('What is this?', open = False):

          gr.Markdown('''
            Sometimes the app will crash due to insufficient GPU memory. If this happens, you can try using the button below to empty the cache. Usually around 12 GB of GPU RAM is needed to safely run the app.

            If that doesn’t work, you’ll have to restart the runtime (`Runtime` > `Restart and run all` in Colab). That’ll take a couple of minutes, but the memory will be new as a daisy.
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
            api_name = 'get-gpu-memory-usage',
          )

          gr.Button('Empty cache', variant='primary').click(
            inputs = None,
            outputs = memory_usage,
            fn = lambda: [
              empty_cache(),
              get_gpu_memory_usage(),
            ][-1],
            api_name = 'empty-cache',
          )

        with gr.Accordion('Dev', open = False, visible = DEV_MODE):

          # Button to kill current process
          gr.Button('Kill current process').click(
            inputs = None,
            outputs = None,
            fn = lambda: subprocess.run( ['kill', '-9', str(os.getpid())] ),
            api_name = 'kill-current-process',
          )

          gr.Markdown('''
            The following input box allows you to execute arbitrary Python code. ⚠️ DON’T USE THIS FEATURE IF YOU DON’T KNOW WHAT YOU’RE DOING! ⚠️
          ''')

          eval_server_code = gr.Textbox(
            label = 'Python code',
            placeholder = 'Shift+Enter for a new line, Enter to run',
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
            inputs = eval_server_code,
            outputs = eval_output,
            fn = lambda code: {
              eval_output: eval( code )
            }
          )

          eval_button.click(**eval_args)
          eval_server_code.submit(
            **eval_args,
            api_name = 'eval-code',
          )

  # TODO: Don't forget to remove this line before publishing the app
  frontend_on_load_url = f'https://cdn.jsdelivr.net/gh/vzakharov/jukebox-webui@{GITHUB_SHA}/frontend-on-load.js'
  with urllib.request.urlopen(frontend_on_load_url) as response:
    frontend_on_load_js = response.read().decode('utf-8')

    try:
      old_frontend_on_load_md5 = frontend_on_load_md5
    except NameError:
      old_frontend_on_load_md5 = None

    frontend_on_load_md5 = hashlib.md5(frontend_on_load_js.encode('utf-8')).hexdigest()
    print(f'Loaded frontend-on-load.js from {response.geturl()}, md5: {frontend_on_load_md5}')

    if frontend_on_load_md5 != old_frontend_on_load_md5:
      print('(New version)')
    else:
      print('(Same version as during the previous run)')

    # print(frontend_on_load_js)

  app.load(
    on_load,
    inputs = [ gr.Textbox(visible=False), gr.Textbox(visible=False), gr.Textbox(visible=False) ],
    outputs = [ 
      UI.project_name, UI.routed_sample_id, UI.artist, UI.genre_dropdown, UI.getting_started_column, UI.separate_tab_warning, UI.separate_tab_link, UI.main_window,
      UI.genre_for_upsampling_left_channel, UI.genre_for_upsampling_center_channel, UI.genre_for_upsampling_right_channel
    ],
    api_name = 'initialize',
    _js = frontend_on_load_js,
    # _js = """
    # // (insert manually for debugging)
    # """,
  )

  # Also load browser's time zone offset on app load
  def set_browser_timezone(offset):
    global browser_timezone

    print('Browser time zone offset:', offset)
    browser_timezone = timezone(timedelta(minutes = -offset))
    print('Browser time zone:', browser_timezone)

  app.load(
    inputs = gr.Number( visible = False ),
    outputs = None,
    _js = '() => [ new Date().getTimezoneOffset() ]',
    fn = set_browser_timezone
  )

  app.launch( share = share_gradio, debug = debug_gradio )