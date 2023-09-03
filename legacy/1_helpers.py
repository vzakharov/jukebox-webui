if True:

  def seconds_to_tokens(sec, sr, prior, chunk_size):
    global hps, raw_to_tokens
    tokens = sec * hps.sr // raw_to_tokens
    tokens = ((tokens // chunk_size) + 1) * chunk_size
    # assert tokens <= prior.n_ctx, 'Choose a shorter generation length to stay within the top prior context'
    tokens = int(tokens)
    return tokens

  def timestamp():
    return datetime.now().astimezone(timezone(timedelta(hours=3))).strftime("%Y%m%d-%H%M%S")

  def write_files(base_filename, display_audio=False, write_wavs=True):
    # Remove .zs for the id
    generation_id = base_filename[:-3]
    j = 0
    available_js=[]
    for i in range(hps.n_samples):
      filename = generation_id
      # Increase j until we find a filename that doesn't exist
      j += 1
      while os.path.exists(f'{filename}-{j}.wav'):
        j += 1
      available_js.append(j)
      if write_wavs:
        filename = f'{filename}-{j}.wav'
        print(f'{filename}:')
        librosa.output.write_wav(filename, x[i], sr=hps.sr)
        if display_audio:
          display(Audio(filename))
      else:
        # Print once that we are not writing wavs
        if i == 0:
          print('Not writing wavs')

    # Save the zs as generation_id-{concated j's}.zs
    base_filename = f'{generation_id}-{",".join(map(str, available_js))}.zs'
    t.save(zs, base_filename)
    return base_filename

  def timestamp():
    return datetime.now().astimezone(timezone(timedelta(hours=3))).strftime("%Y%m%d-%H%M%S")

  # Function to convert a zs file to several wav files (according to the shape of the zs[index]). Default index is 2.
  def zs_to_wavs(id=None, level=2, write_wavs=True, split_into_seconds=None, include_level=True, out={}):
    global vqvae, hps, raw_to_tokens, chunk_size, top_prior
    # if no id is given, use the latest zs formatted as project_name-tmp-yyyy-mm-dd-hh-mm-ss.zs
    if not id:
      filter = f'{hps.name}/tmp/{project_name}-tmp-*.zs'
      print(f'No id given, using latest file matching {filter}')
      results = sorted(glob.glob(filter))
      assert len(results) > 0, f'No files matching {filter}'
      id = results[-1]
      # Remove hps.name and extension (of any length) to get the id
      id = re.sub(f'^{hps.name}/', '', re.sub(r'\.[^.]*$', '', id))
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

    # Remove -1,2,... (or other numerals) from the end of the id
    id = re.sub(r'-\d+$', '', id)

    for i in range(zs[level].shape[0]):
      filename = f'{id}-{i+1}'
      if include_level:
        filename += f' (L{level})'
      filename += '.wav'
      if write_wavs:
        librosa.output.write_wav(f'{hps.name}/{filename}', x[i], sr=hps.sr)
        # display(Audio(f'{hps.name}/{filename}'))
    return x

  # Monitor the folder for new zs files and convert them to wavs when they appear
  def monitor_for_new_zs(immediate=False):
    global hps, project_name
    latest_zs_filename = None

    def get_latest_zs_filename():
      _filter = f'{hps.name}/{project_name}-*.zs'
      # Filter and sort by modification time (descending)
      results = sorted(glob.glob(_filter), key=os.path.getmtime, reverse=True)
      if len(results) > 0:
        return results[0]
      else:
        return None

    print(f'Monitoring {hps.name} for new zs files matching {project_name}-*.zs')

    if not immediate:
      latest_zs_filename = get_latest_zs_filename()
      print(f'Currrent latest zs filename (not processing): {latest_zs_filename}')

    while True:
      time.sleep(1)
      new_latest_zs_filename = get_latest_zs_filename()
      if new_latest_zs_filename != latest_zs_filename:
        print(f'New zs file: {new_latest_zs_filename}')
        zs_to_wavs(new_latest_zs_filename)
        latest_zs_filename = new_latest_zs_filename

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
      # If there's no length, use the whole z
      if piece.get('length'):
        length = seconds_to_tokens(piece['length'], hps.sr, top_prior, chunk_size)
      else:
        length = zs[2].shape[1]
      
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
  #   {'filename': 'nihil3-A-1,2', 'index': 1, 'start': 0, 'length': 49},
  #   {'filename': 'nihil3-A-2-1,2', 'index': 0, 'start': 19, 'length': 3.18}
  # ], 'nihil-A-3,4')


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

  def keep_alive():
    # Keep the Colab session alive
    while True:
      time.sleep(60)
  
  def save_nested(object, dirname, indent = 0):
    path = f'{hps.name}/{dirname}'
    # Try to save the object. If it fails due to AttributeError, save its children while going deeper in the directory structure
    try:
      # Create the directory if it doesn't exist
      if not os.path.exists(path):
        print(f'{"  "*indent}Creating directory {path}')
        os.makedirs(path)
      t.save(object, f'{path}/data.bin')
      print(f'{"  "*indent}Saved {path}/data.bin')
      # Clear the object from memory
      del object
      empty_cache()
    except AttributeError:
      print(f'{"  "*indent}Saving {path} failed, trying to save its children')
      # Delete data.bin in case it was already created
      if os.path.exists(f'{path}/data.bin'):
        os.remove(f'{path}/data.bin')
      for key, value in object.__dict__.items():
        print(f'{"  "*indent}Saving {key}')
        save_nested(value, f'{dirname}/{key}', indent + 1)

  def load_nested(object, dirname, indent = 0):
    path = f'{hps.name}/{dirname}'
    # Try to load the object. If it fails due to FileNotFoundError, load its children while going deeper in the directory structure
    try:
      object = t.load(f'{path}/data.bin')
      print(f'{"  "*indent}Loaded {path}/data.bin')
      return object
    except FileNotFoundError:
      print(f'{"  "*indent}Loading {path} failed, trying to load its children')
      # Scan the directory for children, naming every child after the name of its directory
      for child in os.listdir(path):
        print(f'{"  "*indent}Loading {child}')
        setattr(object, child, load_nested(f'{dirname}/{child}', indent + 1))
      return object