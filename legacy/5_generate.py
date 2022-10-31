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
else:
  print('skipping GENERATE')