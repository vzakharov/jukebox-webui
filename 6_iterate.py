if step == "ITERATE":

  custom_filename = "samael-1,2.zs" #@param{type:'string'}
  my_choice = "2" #@param ["RETRY", "1", "2", "3"]
  continue_generation_in_seconds=3 #@param{type:'number'}
  loop = True #@param{type:'boolean'}

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

  while True:

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

    print(f'Previous zs: {zs_old_filename}')
    zs_filename = write_files(zs_filename)

    if not loop:
      break
    else:
      print('Looping')
else:
  print('skipping ITERATE')