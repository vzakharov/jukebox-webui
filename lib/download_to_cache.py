# Monkey patch load_checkpoint, allowing to load models from arbitrary paths
from jukebox.utils.remote_utils import download


import os


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