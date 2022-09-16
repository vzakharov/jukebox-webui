# !nvidia-smi

#### SETUP ####

if True:

  from google.colab import drive
  drive.mount('/content/gdrive')
  !pip install git+https://github.com/openai/jukebox.git
  import time
  import jukebox
  import torch as t
  import librosa
  import glob
  import os
  import numpy as np
  import re
  from datetime import datetime, timedelta, timezone
  from IPython.display import Audio, display
  from jukebox.make_models import make_vqvae, make_prior, MODELS, make_model
  from jukebox.hparams import Hyperparams, setup_hparams
  from jukebox import sample
  from jukebox.sample import sample_single_window, _sample, \
                            sample_partial_window, upsample, \
                            load_prompts
  from jukebox.utils.dist_utils import setup_dist_from_mpi
  from jukebox.utils.torch_utils import empty_cache
  rank, local_rank, device = setup_dist_from_mpi()

  # Monkey patch sample_single_window to write the zs to a temporary file
  original_sample_single_window = sample.sample_single_window

  last_sampled = None

  def patched_sample_single_window(*args, **kwargs):
    global zs, t, hps, project_name, priors, chunk_size
    zs = original_sample_single_window(*args, **kwargs)

    try:

      # Find the level being sampled
      level = 0
      for i in range(len(zs)):
        if zs[i].size != 0:
          level = i
      
      # Print the number of seconds already generated at this level
      print(f'Generated {seconds_to_tokens(zs[level].shape[1], hps.sr, priors[level], chunk_size) / hps.sr} seconds at level {level}')
      # Print the remaining number of seconds to be generated at this level
      tokens_left = hps.n_ctx - zs[level].shape[1]
      print(f'Remaining: {seconds_to_tokens(tokens_left, hps.sr, priors[level], chunk_size) / hps.sr} seconds')

      if last_sampled:
        # Calculate the time elapsed since the last sample
        time_elapsed = datetime.now() - last_sampled
        # Calculate the time per token
        time_per_token = time_elapsed / hps.n_ctx
        # Calculate the remaining time
        remaining_time = time_per_token * tokens_left
        print(f'Estimated time remaining: {remaining_time}')

      last_sampled = datetime.now()

      # Save with a timestamp
      filename = f'{project_name}-tmp-{datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M-%S")}.zs'
      # If 'tmp' folder doesn't exist, create it
      if not os.path.exists(f'{hps.name}/tmp'):
        os.makedirs(f'{hps.name}/tmp')
      t.save(zs, f'{hps.name}/tmp/{filename}')
      # If there's a file called 'render' in the project folder, render the current zs (zs_to_wavs)
      if os.path.exists(f'{hps.name}/render'):
        print(f'Rendering {filename}...')
        os.remove(f'{hps.name}/render')
        zs_to_wavs(level=level)
        print(f'Done; don’t forget to create a file called “render” to render again.')
    
    except Exception as e:
      print(e)
    
    finally:
      return zs

  sample.sample_single_window = patched_sample_single_window

  model = "5b_lyrics"
  hps = Hyperparams()
  hps.hop_fraction = [.5, .5, .125] 
  hps.levels = 3
  hps.sr = 44100

  chunk_size = 16
  max_batch_size = 3
  lower_batch_size = 16
  lower_level_chunk_size = 32

  raw_to_tokens = 128
  sample_length_in_seconds = 200         #@param{type:'number'}

  hps.sample_length = (int(sample_length_in_seconds*hps.sr)//raw_to_tokens)*raw_to_tokens

  vqvae, *priors = MODELS[model]
  vqvae = make_vqvae(setup_hparams(vqvae, dict(sample_length = hps.sample_length)), device)
  top_prior = make_prior(setup_hparams(priors[-1], dict()), vqvae, device)

  assert hps.sample_length >= top_prior.n_ctx*top_prior.raw_to_tokens, f'Please choose a larger sampling rate'

  zs_old_filename = None
  old_choice = None

#### HELPER FUNCTIONS ####

if True:

  def seconds_to_tokens(sec, sr, prior, chunk_size):
    tokens = sec * hps.sr // prior.raw_to_tokens
    tokens = ((tokens // chunk_size) + 1) * chunk_size
    # assert tokens <= prior.n_ctx, 'Choose a shorter generation length to stay within the top prior context'
    tokens = int(tokens)
    return tokens

  def timestamp():
    return datetime.now().astimezone(timezone(timedelta(hours=3))).strftime("%Y%m%d-%H%M%S")

  def write_files(base_filename):
    # Remove .zs for the id
    generation_id = base_filename[:-3]
    j = 1
    available_js=[]
    for i in range(hps.n_samples):
      filename = generation_id
      # Increase j until we find a filename that doesn't exist
      while os.path.exists(f'{filename}-{j}.wav'):
        j += 1
      available_js.append(j)
      filename = f'{filename}-{j}.wav'
      print(f'{filename}:')
      librosa.output.write_wav(filename, x[i], sr=hps.sr)
      display(Audio(filename))
    # Save the zs as generation_id-{concated j's}.zs
    base_filename = f'{generation_id}-{",".join(map(str, available_js))}.zs'
    t.save(zs, base_filename)
    return base_filename

  def timestamp():
    return datetime.now().astimezone(timezone(timedelta(hours=3))).strftime("%Y%m%d-%H%M%S")

  # Function to write current zs/x to respective files
  def write_files(base_filename):
    # Remove .zs for the id
    generation_id = base_filename[:-3]
    j = 1
    available_js=[]
    for i in range(hps.n_samples):
      filename = generation_id
      # Increase j until we find a filename that doesn't exist
      while os.path.exists(f'{filename}-{j}.wav'):
        j += 1
      available_js.append(j)
      filename = f'{filename}-{j}.wav'
      print(f'{filename}:')
      librosa.output.write_wav(filename, x[i], sr=hps.sr)
      display(Audio(filename))
    # Save the zs as generation_id-{concated j's}.zs
    base_filename = f'{generation_id}-{",".join(map(str, available_js))}.zs'
    t.save(zs, base_filename)
    return base_filename

  # Function to convert a zs file to several wav files (according to the shape of the zs[index]). Default index is 2.
  def zs_to_wavs(id=None, level=2, write_wavs=True, split_into_seconds=None, out={}):
    global vqvae, hps, raw_to_tokens, chunk_size, top_prior
    # if no id is given, use the latest zs formatted as project_name-tmp-yyyy-mm-dd-hh-mm-ss.zs
    if not id:
      filter = f'{hps.name}/{project_name}-tmp-*.zs'
      print(f'No id given, using latest file matching {filter}')
      results = sorted(glob.glob(filter))
      assert len(results) > 0, f'No files matching {filter}'
      id = results[-1]
      # Remove the path and extension (of any length) to get the id
      id = re.sub(r'.*/', '', re.sub(r'\..*$', '', id))
      out['id'] = id
    # if id is of the format hh-mm-ss, add project_name, 'tmp', and current date (yyyy-mm-dd format) to the left
    elif re.match(r'^\d{2}-\d{2}-\d{2}$', id):
      id = f'{project_name}-tmp-{datetime.now().strftime("%Y-%m-%d")}-{id}'
    print(id)
    zs = t.load(f'{hps.name}/{id}.zs')

    if split_into_seconds:

      xs = []
      # Split zs by seconds. Keep a second at the end of each split to overlap with the next split so as to avoid clicks.
      for i in range(0, zs[level].shape[0], split_into_seconds * hps.sr //raw_to_tokens):
        zs_piece = zs[level][i:i+split_into_seconds * hps.sr //raw_to_tokens+1]
        print(f'Generating x for piece {i+1} of shape {zs_piece.shape}')
        x = vqvae.decode([zs_piece], start_level=level, end_level=level+1).cpu().numpy()
        xs.append(x)

      # Merge xs into one long x, making sure to overlap by one second
      x = xs[0]
      for i in range(1, len(xs)):
        # Prepare the overlapping part by gradually fading out the last second of the previous x and fading in the first second of the next x
        faded_out_previous_x = xs[i-1][:, -seconds_to_tokens(1, hps.sr, top_prior, chunk_size):] * np.linspace(1, 0, seconds_to_tokens(1, hps.sr, top_prior, hps.chunk_size))
        faded_in_next_x = xs[i][:, :seconds_to_tokens(1, hps.sr, top_prior, chunk_size)] * np.linspace(0, 1, seconds_to_tokens(1, hps.sr, top_prior, hps.chunk_size))
        # Sum the two parts
        overlap = faded_out_previous_x + faded_in_next_x
        # Concatenate the previous part minus the last second, the overlap, and the next part minus the first second
        x = np.concatenate((x[:, :-seconds_to_tokens(1, hps.sr, top_prior, chunk_size)], overlap, xs[i][:, seconds_to_tokens(1, hps.sr, top_prior, hps.chunk_size):]), axis=1)

      else:
        x = vqvae.decode(zs[level:level+1], start_level=level, end_level=level+1).cpu().numpy()

    for i in range(zs[level].shape[0]):
      filename = f'{id}-{i+1}.wav'
      print(f'{filename}:')
      if write_wavs:
        librosa.output.write_wav(f'{hps.name}/{filename}', x[i], sr=hps.sr)
        display(Audio(f'{hps.name}/{filename}'))
    return x

  # Function to convert wavs to a single wav file by panning the x[index] across the stereo channels
  def wavs_to_stereo(wavs, id, order=None):

    if order:
      # Reorder wavs according to order
      wavs = wavs.take(order, axis=0)

    # Distribute panning evenly from -1 to 1 according to the number of wavs
    # Use linear panning to distribute the wav across the stereo channels
    # Remember that wavs.shape = (n_samples, sample_length, 1)
    stereo = np.matmul(
      np.vstack((np.linspace(-1, 1, wavs.shape[0]), np.linspace(1, -1, wavs.shape[0]))),
      wavs[:, :, 0]
    )
    print(stereo)
    stereo = np.asfortranarray(stereo)
    # Write the stereo wav to a file
    filename = f'{id}-stereo.wav'
    print(f'{filename}:')
    librosa.output.write_wav(f'{hps.name}/{filename}', stereo, sr=hps.sr)
    display(Audio(f'{hps.name}/{filename}'))
    return stereo

  # Function to convert a zs file to a single stereo wav file
  def zs_to_stereo(id=None, level=2, split_into_seconds=None, order=None, out={}):
    wavs = zs_to_wavs(id, level, write_wavs=False, split_into_seconds=split_into_seconds, out=out)
    if not id:
      id = out['id']
    stereo = wavs_to_stereo(wavs, id, order)
    return stereo

  # Function to concatenate two zs files with certain split points (in seconds) and save as wav
  def concat_zs(zs1_filename, z1_index, zs2_filename, z2_index, cut_z1_after_sec, cut_z2_before_sec, filename):
    zs1 = t.load(f'{hps.name}/{zs1_filename}.zs')
    zs2 = t.load(f'{hps.name}/{zs2_filename}.zs')
    cut_z1_after_tokens = seconds_to_tokens(cut_z1_after_sec, hps.sr, top_prior, chunk_size)
    cut_z2_before_tokens = seconds_to_tokens(cut_z2_before_sec, hps.sr, top_prior, chunk_size)
    z1 = zs1[2][z1_index][:cut_z1_after_tokens]
    z2 = zs2[2][z2_index][cut_z2_before_tokens:]
    zs1[2] = z1.repeat(hps.n_samples,1)
    zs2[2] = z2.repeat(hps.n_samples,1)
    z = t.cat([zs1[2], zs2[2]], dim=1)
    # Use empty cuda tensors of size (3, 0) and dtype=torch.int64 for the first two zs
    zs = [t.empty((3,0), dtype=t.int64).cuda(), t.empty((3,0), dtype=t.int64).cuda(), z]
    wav = vqvae.decode(zs[2:], start_level=2).cpu().numpy()
    librosa.output.write_wav(f'{hps.name}/{filename}.wav', wav[0], hps.sr)
    t.save(zs, f'{hps.name}/{filename}.zs')
  
  # Function to concatenate an arbitrary number of zs files with certain split points (in seconds) and save as wav
  def concat_zs_multiple(pieces, filename):
    # pieces is a list of dicts with the following keys:
    #   filename: the filename of the zs file
    #   index: the index of the z to use
    #   start: the start time to cut from, default 0
    #   length: the length of the z to use, default None (use the whole z)

    combined_z = None
    for piece in pieces:
      zs = t.load(f'{hps.name}/{piece["filename"]}.zs')
      start = seconds_to_tokens(piece.get('start', 0), hps.sr, top_prior, chunk_size)
      length = seconds_to_tokens(piece.get('length', None), hps.sr, top_prior, chunk_size)
      z = zs[2][piece['index']][start:start+length]
      print(f'z: {z.shape}')
      if combined_z is None:
        combined_z = z
      else:
        combined_z = t.cat([combined_z, z], dim=0)
        print(f'combined_z: {combined_z.shape}')
    zs = [
      t.empty(hps.n_samples, 0, dtype=t.int64).cuda(),
      t.empty(hps.n_samples, 0, dtype=t.int64).cuda(),
      combined_z.repeat(hps.n_samples, 1)
    ]
    t.save(zs, f'{hps.name}/{filename}.zs')
    wav = vqvae.decode(zs[2:], start_level=2).cpu().numpy()
    librosa.output.write_wav(f'{hps.name}/{filename}.wav', wav[0], hps.sr)

  # Example usage to combine the following files:
  #   nihil-0-0.zs at index 0, start at 0, length 38
  #   nihil-5-1-1-1,2.zs at index 0, start at 1, length 10.719
  #   nihil-A-2-1-2-1-2-4-2-1,2.zs at index 1, start at 1.688, length 42.280
  #   nihil-90s-1-1-1-1-2-2-1-2-2-1-2-1-1,2 at index 0, start at 23.953, length 89.356
  # concat_zs_multiple([
  #   {'filename': 'nihil-0-0', 'index': 0, 'start': 0, 'length': 38},
  #   {'filename': 'nihil-5-1-1-1,2', 'index': 0, 'start': 1, 'length': 10.719},
  #   {'filename': 'nihil-A-2-1-2-1-2-4-2-1,2', 'index': 1, 'start': 1.688, 'length': 42.280},
  #   {'filename': 'nihil-90s-1-1-1-1-2-2-1-2-2-1-2-1-1,2', 'index': 0, 'start': 23.953, 'length': 89.356},
  # ], 'nihil-combined')


  def check_gpu_usage(object = None, key = None, depth = 0, processed = []):
    if object == 'globals':
      object = globals()

    # Exit if it is a dict or list and has already been processed
    if (isinstance(object, dict) or isinstance(object, list)):
      if processed.count(object) > 0:
        return

      processed.append(object)
      
      if isinstance(object, list):
        for value, i in enumerate(object):
          check_gpu_usage(value, i, depth + 1, processed)
      elif isinstance(object, dict):
        for key, value in object.items():
          check_gpu_usage(value, key, depth + 1, processed)

    elif t.is_tensor(object):  
      print(f'{"  "*(depth+2)}{key}: {object.size()}')

#### PARAMETERS ####

if True:

  #@title  { run: "auto" }
  project_name = 'nihil' #@param{type:'string'} 
  hps.name = '/content/gdrive/My Drive/AI music/'+project_name

  artist = "Within Temptation" #@param{type:'string'}
  genre = "Metal" #@param{type:'string'}

  lyrics = "I am the void, the empty space; / The nothingness that surrounds you. / I am the silence between the notes; / The darkness between the stars. /  / I am the end and the beginning; / The alpha and the omega. / I am the void, the abyss; / The nothingness that consumes you. /  / Nihil, / The terror of the blank page; / Nihil, / The freedom of an unmade world. / Nihil, / The peace of an empty mind; / Nihil, / The nothing that is everything." #@param{type:'string'}
  lyrics = lyrics.replace(' / ', '\n')

  print(lyrics)


  sample_length_in_seconds = 120         #@param{type:'number'}
  hps.sample_length = (int(sample_length_in_seconds*hps.sr)//top_prior.raw_to_tokens)*top_prior.raw_to_tokens
  assert hps.sample_length >= top_prior.n_ctx*top_prior.raw_to_tokens, f'Please choose a larger sampling rate'

  metas = [dict(artist = artist,
              genre = genre,
              total_length = hps.sample_length,
              offset = 0,
              lyrics = lyrics,
              ),
            ] * hps.n_samples
  labels = top_prior.labeller.get_batch_labels(metas, 'cuda')


  n_samples = 3 #@param {type:"integer"}

  sampling_temperature = .98 #@param{type:'number'}
  sampling_kwargs = dict(temp=sampling_temperature, fp16=True, max_batch_size=lower_batch_size,
                          chunk_size=lower_level_chunk_size)

  primed = False #@param{type:'boolean'}
  prime_suffix = 'in' #@param{type:'string'}

  codes_file=None
  audio_file=None
  prompt_length_in_seconds=10 #@param{type:'number'}

  mode = 'ancestral'

  if primed:
    audio_file = hps.name+'/'+project_name+'-'+prime_suffix+'.wav'
    mode = 'primed'

  sample_hps = Hyperparams(dict(mode=mode, codes_file=codes_file, audio_file=audio_file, prompt_length_in_seconds=prompt_length_in_seconds))

#### CHOOSE STEP ####

step = 'GENERATE' #@param ['GENERATE', 'ITERATE', 'UPSAMPLE']

#### GENERATE ####

if step == 'GENERATE':

  if not os.path.exists(hps.name):
    os.makedirs(hps.name)

  initial_generation_in_seconds = 3 #@param{type:'number'}
  tokens_to_sample = seconds_to_tokens(initial_generation_in_seconds, hps.sr, top_prior, chunk_size)
  print(tokens_to_sample)


  if primed:
    assert sample_hps.audio_file is not None
    audio_files = sample_hps.audio_file.split(',')
    duration = (int(sample_hps.prompt_length_in_seconds*hps.sr)//top_prior.raw_to_tokens)*top_prior.raw_to_tokens
    x = load_prompts(audio_files, duration, hps)
    zs = top_prior.encode(x, start_level=0, end_level=len(priors), bs_chunks=x.shape[0])
  else:
    zs=[t.zeros(hps.n_samples,0,dtype=t.long, device='cuda') for _ in range(3)]

  zs = sample_partial_window(zs, labels, sampling_kwargs, 2, top_prior, tokens_to_sample, hps)

  x = vqvae.decode(zs[2:], start_level=2).cpu().numpy()

  zs_filename =  f'{hps.name}/{project_name}.zs'

  zs_filename = write_files(zs_filename)

"""## Iterate"""

if creation_mode == "Iterate":

  custom_filename = "samael-1,2.zs" #@param{type:'string'}
  my_choice = "2" #@param ["RETRY", "1", "2", "3"]
  continue_generation_in_seconds=3 #@param{type:'number'}    

  retry = my_choice == 'RETRY'
  zs_old_filename = None
  old_choice = None
  assert zs_old_filename is not None and old_choice is not None or not retry, 'Can’t retry without a previous generation'

  if retry:
    assert not custom_filename, 'Can’t use custom filename without choice'
    custom_filename = zs_old_filename
    my_choice = old_choice
  else:
    my_choice = int(my_choice)-1

  if custom_filename:
    if not retry:
      # If only the filename is given (without path), add the path
      if '/' not in custom_filename:
        custom_filename=f'{hps.name}/{custom_filename}'
      # If extension is missing, add .zs
      if not custom_filename.endswith('.zs'):
        custom_filename += '.zs'
    zs = t.load(f'{custom_filename}')
    zs_filename = custom_filename

  z = zs[2][my_choice]

  cut_before_seconds = None #@param{type:'number'}
  cut_after_seconds = None #@param{type:'number'}
  if cut_after_seconds:
    cut_after_tokens = seconds_to_tokens(cut_after_seconds, hps.sr, top_prior, chunk_size)
    z = z[:cut_after_tokens]
  if cut_before_seconds:
    cut_before_tokens = seconds_to_tokens(cut_before_seconds, hps.sr, top_prior, chunk_size)
    z = z[cut_before_tokens:]

  print(f'Using {zs_filename}')
  print(f'  Choice {my_choice + 1}')
  zs[2]=z.repeat(hps.n_samples,1)

  zs_old_filename=zs_filename
  old_choice=my_choice

  # Take the three numbers after the last hyphen in the filename (excluding the extension) and split them by comma
  splitted = zs_filename.split('.')[0].split('-')
  variations = splitted[-1].split(',')
  # Add the current choice to the filename
  zs_filename = f'{"-".join(splitted[:-1])}-{variations[my_choice]}.zs'


  empty_cache()


  tokens_to_sample = seconds_to_tokens(continue_generation_in_seconds, hps.sr, top_prior, chunk_size)
  total_seconds = zs[2].shape[1]//344 + continue_generation_in_seconds
  print(total_seconds)
  # if sampling_kwargs doesn't have a 'max_batch_size' key, it will use the default value
  if not 'max_batch_size' in sampling_kwargs:
    sampling_kwargs['max_batch_size']=16
  print(sampling_kwargs)
  print(hps)

  zs = sample_partial_window(zs, labels, sampling_kwargs, 2, top_prior, tokens_to_sample, hps)
  x = vqvae.decode(zs[2:], start_level=2).cpu().numpy()
  empty_cache()

  zs_old_filename=zs_filename
  old_choice=my_choice
  print(f'Previous zs: {zs_old_filename}')
  zs_filename = write_files(zs_filename)

"""## Upsample"""

if creation_mode == "Upsample":

  choice = 0
  select_best_sample = True  # Set false if you want to upsample all your samples 
                            # upsampling sometimes yields subtly different results on multiple runs,
                            # so this way you can choose your favorite upsampling

  if select_best_sample:
    zs[2]=zs[2][choice].repeat(zs[2].shape[0],1)

  t.save(zs, 'zs-top-level-final.t')

  """Note: If you are using a CoLab hosted runtime on the free tier, you may want to download this zs-top-level-final.t file, and then restart an instance and load it in the next cell.  The free tier will last a maximum of 12 hours, and the upsampling stage can take many hours, depending on how long a sample you have generated."""

  if True:
    zs = t.load('zs-checkpoint2 (6).t')
    # print(zs[2])
    # print(zs[2].shape[1])
    # zs[2] = zs[2][:1]
    # print(zs[2])
    # print(zs[2].shape[1])

  assert zs[2].shape[1]>=2048, f'Please first generate at least 2048 tokens at the top level, currently you have {zs[2].shape[1]}'
  hps.sample_length = zs[2].shape[1]*128 #top_prior.raw_to_tokens

  # Set this False if you are on a local machine that has enough memory (this allows you to do the
  # lyrics alignment visualization). For a hosted runtime, we'll need to go ahead and delete the top_prior
  # if you are using the 5b_lyrics model.
  if True:
    del top_prior
    empty_cache()
    top_prior=None

  upsamplers = [make_prior(setup_hparams(prior, dict()), vqvae, 'cpu') for prior in priors[:-1]]

  sampling_kwargs = [dict(temp=.99, fp16=True, max_batch_size=16, chunk_size=32),
                      dict(temp=0.99, fp16=True, max_batch_size=16, chunk_size=32),
                      None]

  if type(labels)==dict:
    labels = [prior.labeller.get_batch_labels(metas, 'cuda') for prior in upsamplers] + [labels]

  """This next step upsamples 2 levels.  The level_1 samples will be available after around one hour (depending on the length of your sample) and are saved under {hps.name}/level_0/item_0.wav, while the fully upsampled level_0 will likely take 4-12 hours. You can access the wav files down below, or using the "Files" panel at the left of this CoLab.

  (Please note, if you are using this CoLab on Google's free tier, you may want to download intermediate steps as the connection will last for a maximum 12 hours.)
  """

  # print(hps)

  print(sampling_kwargs)
  zs = upsample(zs, labels, sampling_kwargs, [*upsamplers, top_prior], hps)

  Audio(f'{hps.name}/level_0/item_0.wav')