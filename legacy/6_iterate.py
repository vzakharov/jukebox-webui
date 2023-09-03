if step == "ITERATE":

  custom_filename = "artemis-P2J-2-3-1" #@param{type:'string'}
  my_choice = "1" #@param ["RETRY", "1", "2", "3"]

  cut_before_seconds = 0 #@param{type:'number'}
  cut_after_seconds = 120.75 #@param{type:'number'}


  continue_generation_in_seconds=2 #@param{type:'number'}
  loop_times = 3 #@param{type:'number'}
  loop_tree = False #@param{type:'boolean'}
  write_wavs = True #@param{type:'boolean'}
  display_audio = False #@param{type:'boolean'}
  delete_siblings = False #@param{type:'boolean'}
  delete_children = False #@param{type:'boolean'}

  retry = my_choice == 'RETRY'
  zs_old_filename = None
  old_choice = None
  assert zs_old_filename is not None and old_choice is not None or not retry, 'Can’t retry without a previous generation'

  zs_stem = None
  zs_stem_filename = None
  random_choice = False
  prevent_cutting = False

  while loop_times > 0:

    print(f'{loop_times} loops left')
    loop_times -= 1

    if not random_choice:
      if retry:
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

    if random_choice:
      # Pick random index from zs[2]'s first dimension
      my_choice = np.random.randint(zs[2].shape[0])

    z = zs[2][my_choice]

    if cut_after_seconds:
      if prevent_cutting:
        print('Not the stem, won’t cut')
      else:
        print(f'Cutting after {cut_after_seconds} seconds')
        cut_after_tokens = seconds_to_tokens(cut_after_seconds, hps.sr, top_prior, chunk_size)
        z = z[:cut_after_tokens]
    if cut_before_seconds:
      print(f'Cutting before {cut_before_seconds} seconds')
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
    zs_base_filename = f'{"-".join(splitted[:-1])}-{variations[my_choice]}.zs'

    if not zs_stem:
      zs_stem = ''.join(zs_base_filename.split('.')[:-1])
      zs_stem_filename = zs_filename
      # Remove hps_name/ from zs_stem
      zs_stem = zs_stem[len(hps.name)+1:]
      variation = variations[my_choice]
      print(f'Using {zs_stem} as the stem')

      if delete_siblings or delete_children:
        # Assert that they are not both True, otherwise suggest to choose one
        assert not (delete_siblings and delete_children), 'Choose either delete_siblings or delete_children but not both'
        if delete_siblings:
          prefix_for_deletion = zs_stem.rsplit('-', 1)[0]
        else:
          prefix_for_deletion = zs_stem
        # Wait for input to avoid accidental deletion
        input(f'Press Enter to delete {prefix_for_deletion}*.wav (except {zs_stem}.wav)')
        # Create /deleted if it doesn’t exist
        if not os.path.exists(f'{hps.name}/deleted'):
          os.makedirs(f'{hps.name}/deleted')
        # Move all wavs that start with prefix_for_deletion to /deleted (but not the stem itself) with a timestamp suffix
        for f in glob.glob(f'{hps.name}/{prefix_for_deletion}-*.wav'):
          if f != f'{hps.name}/{zs_stem}.wav':
            os.rename(f, f'{hps.name}/deleted/{f.split("/")[-1].split(".")[0]} ({timestamp()}).wav')


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
    write_wavs = True

    try:
      x = vqvae.decode(zs[2:], start_level=2).cpu().numpy()
    except RuntimeError as e:
      write_files(zs_base_filename, write_wavs=False)
      print(f'Out of memory. Saved {zs_base_filename} and exiting')
      break

    empty_cache()

    zs_filename = write_files(zs_base_filename, display_audio=display_audio, write_wavs=write_wavs)

    # Delete all large variables and free memory
    del z, x
    empty_cache()

    # If there's a directory named 'stop', stop the loop
    if os.path.isdir(f'{hps.name}/stop'):
      print('Found stop folder, stopping')
      # delete the stop folder
      !rm -r '{hps.name}/stop'
      break

    # If loop_tree is False, just rerun the same choice
    if not loop_tree:
      retry = True
    else:
      # Find all files starting with the same name as f'{zs_stem}-' and have the .zs extension
      files = [f for f in os.listdir(hps.name) if f.startswith(zs_stem) and f.endswith('.zs')]
      if len(files) >= 1:
        print('Available variations:')
        for i, f in enumerate(files):
          print(f'{i+1}: {f}')
        # Pick a random file
        custom_filename = np.random.choice(files)
        if custom_filename == zs_filename:
          my_choice = variation + 1
          prevent_cutting = False
        else:
          random_choice = True
          prevent_cutting = True
      else:
        print('No more variations')
        break

  # Play a ding sound from some public domain sound effects
  display(Audio('https://cdn.freesound.org/previews/173/173932_3229685-lq.mp3', autoplay=True))
  

else:
  print('skipping ITERATE')