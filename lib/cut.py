from lib.navigation.utils import get_project_name_from_sample_id, get_zs
from lib.utils import seconds_to_tokens

import torch as t

def cut_z(z, specs, level):
  # possible specs:
  # start-end -- cuts out the specified range (in seconds), either can be omitted to cut from the start or to the end, the dash can be omitted to cut from the specified time to the end
  # start-end+sample_id@start-end -- takes the specified range from the specified sample and adds it instead of the cut-out range
  # start-end+start-end -- same, but takes the specified range from the current sample
  # +sample_id@start-end -- adds the specified range from the specified sample to the end of the current sample
  # +start-end -- keeps just the specified range from the current sample (i.e. the opposite of start-end)
  # Any whitespaces are ignored
  # All of these can be combined with a comma to cut out multiple ranges
  specs = specs.replace(' ', '')

  print(f'z shape before cut: {z.shape}')

  for spec in specs.split(','):

    remove, add = spec.split('+') if '+' in spec else (spec, None)

    out_z = z[:, :0]

    if remove:

      # The removed interval is a string of format 'start-end' or just 'start'. In the latter case, end is assumed to be the end of the sample
      remove_start, remove_end = remove.split('-') if '-' in remove else (remove, None)

      # Hidden spec: if either start or end start with a '<', the corresponding value is taken from the end of the sample (i.e. we just negate the value, i.e. replace '<' with '-')
      # ("<" because "-" is already used for specifying the interval. It also looks like a backwards arrow which is a good visual cue for this)
      remove_start, remove_end = [ s and s.replace('<', '-') for s in (remove_start, remove_end) ]

      # If start or end is empty, it means the interval starts at the beginning or ends at the end
      remove_start = seconds_to_tokens(float(remove_start), level) if remove_start else 0

      # If remove_start is more than the length of the sample, we just return an empty sample
      # (We don't need to see the add part, because it's not going to be added to anything. The only exception is if no remove part is specified, but in that case this part of the code is not executed anyway)
      if remove_start >= z.shape[1]:
        print(f'Warning: remove_start ({remove_start}) is more than the length of the sample ({z.shape[1]}) for level {level}. Returning an empty sample.')
        break

      remove_end = seconds_to_tokens(float(remove_end), level) if remove_end else z.shape[1]
      print(f'Cutting out {remove} (tokens {remove_start}-{remove_end})')

      out_z = t.cat([out_z, z[:, :remove_start]], dim=1)
      print(f'out_z shape: {out_z.shape}')

    # For the added interval, both start and end are required (but sample_id is optional, and defaults to the current sample)
    if add:
      add_sample_id, add = add.split('@') if '@' in add else (None, add)
      add_start, add_end = add.split('-')
      add_start = seconds_to_tokens(float(add_start), level)
      add_end = seconds_to_tokens(float(add_end), level)
      print(f'Adding {add} (tokens {add_start}-{add_end}) from { add_sample_id or "current sample" }')

      if add_sample_id:
        add_z = get_zs(get_project_name_from_sample_id(add_sample_id), add_sample_id)[level]
        # If no remove was specified, add the entire original sample (before we add the part from the other sample)
        if not remove:
          out_z = z
      else:
        add_z = z
        # (In this case we don't add the original sample, because we just want to keep the specified range)

      out_z = t.cat([out_z, add_z[:, add_start:add_end]], dim=1)
      print(f'out_z shape: {out_z.shape}')

    if remove:
      # If we added anything, and its end was after the end of the original sample, we break
      # This is needed for cases when we add a part that hasn't been upsampled yet, so it would be added for the low-quality level, but not for the high-quality level (at least partially)
      # In this case, we don't want to add the rest of the original sample, because then we would have a "hole" in the high-quality level, which will make further upsampling impossible
      if add and add_end > z.shape[1]:
        print(f'Warning: add_end ({add_end}) is more than the length of the sample ({z.shape[1]}) for level {level}. Breaking before adding the rest of the original sample.')
        break

      print(f'Adding the rest of the sample (tokens {remove_end}-{z.shape[1]})')
      out_z = t.cat([out_z, z[:, remove_end:]], dim = 1)
      print(f'out_z shape: {out_z.shape}')

    z = out_z

  print(f'z shape after cut: {out_z.shape}')
  return z

def cut_zs(zs, specs):
  return [ cut_z(zs[level], specs, level) for level in range(len(zs)) ]

def cut_audio(project_name, sample_id, interval):
  backup_sample(project_name, sample_id)
  zs = get_zs(project_name, sample_id)
  zs = cut_zs(zs, interval)
  save_zs(zs, project_name, sample_id)
  return ''