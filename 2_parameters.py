if True:

  #@title  { run: "auto" }
  project_name = 'nihil' #@param{type:'string'} 
  hps.name = '/content/gdrive/My Drive/AI music/'+project_name

  artist = "Within Temptation" #@param{type:'string'}
  genre = "Metal" #@param{type:'string'}

  lyrics = "I am the void, the empty space; / The nothingness that surrounds you. / I am the silence between the notes; / The darkness between the stars. /  / I am the end and the beginning; / The alpha and the omega. / I am the void, the abyss; / The nothingness that consumes you. /  / Nihil, / The terror of the blank page; / Nihil, / The freedom of an unmade world. / Nihil, / The peace of an empty mind; / Nihil, / The nothing that is everything." #@param{type:'string'}
  lyrics = lyrics.replace(' / ', '\n')

  print(lyrics)

  n_samples = 3 #@param {type:"integer"}
  hps.n_samples = n_samples

  metas = [dict(artist = artist,
              genre = genre,
              total_length = hps.sample_length,
              offset = 0,
              lyrics = lyrics,
              ),
            ] * n_samples
  labels = top_prior.labeller.get_batch_labels(metas, 'cuda')


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