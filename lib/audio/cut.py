from lib.navigation.zs import get_zs, save_zs
from lib.navigation.utils import backup_sample

from .cut_single_spec import cut_single_spec


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
    z, out_z, should_break = cut_single_spec(z, spec, level)
    if should_break:
      break

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