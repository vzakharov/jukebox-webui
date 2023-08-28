from lib.navigation.get_custom_parents import get_custom_parents
from lib.navigation.utils import is_ancestor
from lib.upsampling.Upsampling import Upsampling
from main import load_top_prior, sample_id_to_restart_upsampling_with
from lib.upsampling.request_to_stop_upsampling import request_to_stop_upsampling
from lib.upsampling.start_upsampling import start_upsampling

def restart_upsampling(sample_id, even_if_no_labels = False, even_if_not_ancestor = False):

  global sample_id_to_restart_upsampling_with

  get_custom_parents(Upsampling.project, force_reload = True)

  if Upsampling.running:
    print('Upsampling is already running; stopping & waiting for it to finish to restart')
    request_to_stop_upsampling()
    sample_id_to_restart_upsampling_with = sample_id
    return

  assert not Upsampling.running, 'Upsampling is already running. Use stop_upsampling() to stop it and wait for the current window to finish.'

  assert Upsampling.labels or even_if_no_labels, 'Upsampling.labels is empty, cannot restart. If you want to restart anyway, set even_if_no_labels to True.'
  if not Upsampling.labels:
    load_top_prior()
    # (We deleted the top_prior object in start_upsampling, so we need to reload it to recalculate the labels)

  assert even_if_not_ancestor or is_ancestor(Upsampling.project, Upsampling.sample_id, sample_id), 'Cannot restart upsampling with a sample that is not a descendant of the currently upsampled sample. If you really want to do this, set even_if_not_ancestor to True.'

  start_upsampling(Upsampling.project, sample_id, Upsampling.metas[0]['artist'], Upsampling.metas[0]['lyrics'], *[ meta['genre'] for meta in Upsampling.metas ])
  # (Note that the metas don't do anything here, as we're already using the calculated labels. Keeping for future cases where we might want to restart with different metas.)
  print('Warning: Using the same labels as before. If you want to restart with different labels, you need to set Upsampling.labels to None before calling restart_upsampling.')