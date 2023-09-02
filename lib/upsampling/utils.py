from lib.navigation.zs import get_zs
from lib.navigation.get_parent import get_parent
from main import keep_upsampling_after_restart

def get_levels(zs):

  levels = []
  for i in range(3):
    if zs[i].shape[1] == 0:
      # print(f'Level {i} is empty, skipping')
      pass
    else:
      # We also need to make sure that, if it's not level 2, there are exactly 3 samples in the tensor
      # Otherwise it's a primed sample, not the one we created during upsampling
      # I agree this is a bit hacky; in the future we need to make sure that the primed samples are not saved for levels other than 2
      # But for backwards compatibility, we need to keep this check
      if i != 2 and zs[i].shape[0] != 3:
        # print(f"Level {i}'s tensor has {z[i].shape[0]} samples, not 3, skipping")
        pass
      else:
        levels.append(i)

  return levels

def is_upsampled(zs):
  # Yes if there are at least 2 levels
  return len(get_levels(zs)) >= 2

def get_first_upsampled_ancestor_zs(project_name, sample_id):
  zs = get_zs(project_name, sample_id)
  # print(f'Looking for the first upsampled ancestor of {sample_id}')
  if is_upsampled(zs):
    print(f'Found upsampled ancestor: {sample_id}')
    return zs
  else:
    parent = get_parent(project_name, sample_id)
    if parent:
      return get_first_upsampled_ancestor_zs(project_name, parent)
    else:
      print(f'No upsampled ancestor found for {sample_id}')
      return None

def set_keep_upsampling_after_restart():
  global keep_upsampling_after_restart

  keep_upsampling_after_restart = True