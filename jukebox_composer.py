# -*- coding: utf-8 -*-
"""Jukebox Composer

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1PR3DieSKoJ0obbSVYs7tCLC3kPgZ-3qU
"""

# !nvidia-smi

from google.colab import drive
drive.mount('/content/gdrive')
!pip install git+https://github.com/openai/jukebox.git
import time
import jukebox
import torch as t
import librosa
import os
from datetime import datetime, timedelta
from IPython.display import Audio, display
from jukebox.make_models import make_vqvae, make_prior, MODELS, make_model
from jukebox.hparams import Hyperparams, setup_hparams
from jukebox.sample import sample_single_window, _sample, \
                           sample_partial_window, upsample, \
                           load_prompts
from jukebox.utils.dist_utils import setup_dist_from_mpi
from jukebox.utils.torch_utils import empty_cache
rank, local_rank, device = setup_dist_from_mpi()

model = "5b_lyrics" # @param["1b_lyrics", "5b_lyrics"]
hps = Hyperparams()
hps.sr = 44100
project_name = 'erebos' #@param{type:'string'}
n_samples = 3 #@param {type:"integer"}
hps.n_samples = n_samples
# Specifies the directory to save the sample in.
# We set this to the Google Drive mount point.
hps.name = '/content/gdrive/My Drive/AI music/'+project_name
hps.sample_length = 1048576 if model in ('5b', '5b_lyrics') else 786432 
chunk_size = 16 if model in ('5b', '5b_lyrics') else 32
max_batch_size = 3 if model in ('5b', '5b_lyrics') else 16
hps.hop_fraction = [.5, .5, .125] 
hps.levels = 3

# The default mode of operation.
# Creates songs based on artist and genre conditioning.
mode = "ancestral" #@param ["ancestral", "primed"]
prime_suffix = 'in' #@param{type:'string'}
if mode == 'ancestral':
  codes_file=None
  audio_file=None
  prompt_length_in_seconds=None
else:
# Prime song creation using an arbitrary audio sample.
  codes_file=None
  # Specify an audio file here.
  audio_file = hps.name+'/'+project_name+'-'+prime_suffix+'.wav'
  print(audio_file)
  # Specify how many seconds of audio to prime on.
  prompt_length_in_seconds=10 #@param{type:'number'}
sample_hps = Hyperparams(dict(mode=mode, codes_file=codes_file, audio_file=audio_file, prompt_length_in_seconds=prompt_length_in_seconds))

sample_length_in_seconds = 300         #@param{type:'number'}
raw_to_tokens = 128 #@param{type:'integer'}
n_ctx = 8192 #@param{type:'integer'}
hps.sample_length = (int(sample_length_in_seconds*hps.sr)//raw_to_tokens)*raw_to_tokens
assert hps.sample_length >= n_ctx*raw_to_tokens, f'Please choose a larger sampling rate'

def seconds_to_tokens(sec, sr, prior, chunk_size):
  tokens = sec * hps.sr // prior.raw_to_tokens
  tokens = ((tokens // chunk_size) + 1) * chunk_size
  assert tokens <= prior.n_ctx, 'Choose a shorter generation length to stay within the top prior context'
  return tokens

if not os.path.exists(hps.name):
    os.makedirs(hps.name)

def timestamp():
  return datetime.now().astimezone(timezone(timedelta(hours=3))).strftime("%Y%m%d-%H%M%S")

vqvae, *priors = MODELS[model]
vqvae = make_vqvae(setup_hparams(vqvae, dict(sample_length = hps.sample_length)), device)
top_prior = make_prior(setup_hparams(priors[-1], dict()), vqvae, device)

assert n_ctx == top_prior.n_ctx, f'Please set n_ctx to {top_prior.n_ctx} and rerun the parameters cell'
assert raw_to_tokens == top_prior.raw_to_tokens, f'Please set n_ctx to {top_prior.raw_to_tokens} and rerun the parameters cell'

# Note: Metas can contain different prompts per sample.
# By default, all samples use the same prompt.
artist = "Five Finger Death Punch" #@param{type:'string'}
genre = "Metalcore" #@param{type:'string'}
lyrics = "You are the shadow in the corner of our eye; / The dark passenger that lurks beneath our skin. / We try so hard to keep you locked away, / But you break free and come to claim what\u2019s yours. /  / Erebos, / The god of darkness; / Erebos, / The lord of shadows; / Erebos, / The bringer of nightmares; / Erebos, / You show us who we really are. /  / Some say you are the devil, / But there is divinity in your darkness. / You are the force that drives creation; / The chaos that gives birth to stars." #@param{type:'string'}
lyrics = lyrics.replace(' / ', '\n')
print(lyrics)
metas = [dict(artist = artist,
            genre = genre,
            total_length = hps.sample_length,
            offset = 0,
            lyrics = lyrics,
            ),
          ] * hps.n_samples
labels = top_prior.labeller.get_batch_labels(metas, 'cuda')

"""## Generate"""

initial_generation_in_seconds = 3 #@param{type:'number'}
tokens_to_sample = seconds_to_tokens(initial_generation_in_seconds, hps.sr, top_prior, chunk_size)
print(tokens_to_sample)

sampling_temperature = .98 #@param{type:'number'}

lower_batch_size = 16
max_batch_size = 3 if model in ('5b', '5b_lyrics') else 16
lower_level_chunk_size = 32
chunk_size = 16 if model in ('5b', '5b_lyrics') else 32
sampling_kwargs = dict(temp=sampling_temperature, fp16=True, max_batch_size=lower_batch_size,
                        chunk_size=lower_level_chunk_size)

if sample_hps.mode == 'ancestral':
  zs=[t.zeros(hps.n_samples,0,dtype=t.long, device='cuda') for _ in range(3)]
  zs=sample_partial_window(zs, labels, sampling_kwargs, 2, top_prior, tokens_to_sample, hps)
elif sample_hps.mode == 'primed':
  assert sample_hps.audio_file is not None
  audio_files = sample_hps.audio_file.split(',')
  duration = (int(sample_hps.prompt_length_in_seconds*hps.sr)//top_prior.raw_to_tokens)*top_prior.raw_to_tokens
  x = load_prompts(audio_files, duration, hps)
  zs = top_prior.encode(x, start_level=0, end_level=len(priors), bs_chunks=x.shape[0])
  zs = sample_partial_window(zs, labels, sampling_kwargs, 2, top_prior, tokens_to_sample, hps)

x = vqvae.decode(zs[2:], start_level=2).cpu().numpy()

def get_generation_id(seconds):
  return f'{hps.name}/{project_name}-{str(seconds).zfill(3)}s-{timestamp()}'

def write_files(seconds):
  generation_id = get_generation_id(seconds)
  zs_filename=f'{generation_id}-zs.t'
  t.save(zs, zs_filename)

  for i in range(hps.n_samples):
    filename = f'{generation_id}_{i+1}.wav'
    librosa.output.write_wav(filename, x[i], sr=44100)
    print(filename)
    display(Audio(filename))

write_files(initial_generation_in_seconds)

"""## Iterate"""

custom_filename = "erebos-006s-20220912-125702-zs" #@param{type:'string'}
my_choice = "3" #@param ["RETRY", "1", "2", "3"]
continue_generation_in_seconds=3 #@param{type:'number'}

retry = my_choice == 'RETRY'
if retry:
  assert not custom_filename, 'Can’t use custom filename without choice'
  custom_filename = zs_old_filename
  my_choice = old_choice
else:
  my_choice = int(my_choice)-1

if custom_filename:
  if not retry:
    custom_filename=f'{hps.name}/{custom_filename}.t'
  print(f'Using {custom_filename}')
  zs = t.load(f'{custom_filename}')
  # If the number of samples in zs doesn't match the number of samples in the model, we need to reshape zs
  if zs[2].shape[0] > hps.n_samples:
    print(f'Reshaping zs from {zs[2].shape[0]} to {hps.n_samples}')
    zs[2] = zs[2][:hps.n_samples]
  zs_filename = custom_filename

print(f'Using choice {my_choice + 1}')
zs[2]=zs[2][my_choice].repeat(hps.n_samples,1)

empty_cache()


tokens_to_sample = seconds_to_tokens(continue_generation_in_seconds, hps.sr, top_prior, chunk_size)
total_seconds = zs[2].shape[1]//344 + continue_generation_in_seconds
print(total_seconds)
# sampling_kwargs['max_batch_size']=16
print(sampling_kwargs)
print(hps)

zs = sample_partial_window(zs, labels, sampling_kwargs, 2, top_prior, tokens_to_sample, hps)
x = vqvae.decode(zs[2:], start_level=2).cpu().numpy()
empty_cache()

zs_old_filename=zs_filename
old_choice=my_choice
print(f'Previous zs: {zs_old_filename}')
write_files(total_seconds)

"""# Upsample Co-Composition to Higher Audio Quality

Choose your favorite sample from your latest group of generations.  (If you haven't already gone through the Co-Composition block, make sure to do that first so you have a generation to upsample).
"""

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