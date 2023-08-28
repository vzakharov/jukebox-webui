from lib.navigation.get_sample import get_sample
from lib.navigation.utils import get_siblings
from lib.ui.UI import UI


def get_sibling_samples(project_name, sample_id, cut_out, last_n_sec, upsample_rendering, combine_levels, invert_center):
  print(f'Updating sibling samples for {sample_id}...')
  sibling_files = []
  for sibling_id in get_siblings(project_name, sample_id):
    if sibling_id == sample_id:
      continue
    sibling_sample = get_sample(project_name, sibling_id, cut_out, last_n_sec, upsample_rendering, combine_levels, invert_center)
    sibling_sample_files = sibling_sample[UI.current_chunks]
    # breakpoint()
    print(f'Adding sibling {sibling_id} with files {sibling_sample_files}')
    sibling_files.extend(sibling_sample_files)
    print(f'Totally {len(sibling_files)} sibling files')
  return {
    UI.sibling_chunks: sibling_files
  }