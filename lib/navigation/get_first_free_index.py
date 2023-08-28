from lib.navigation.get_children import get_children


def get_first_free_index(project_name, parent_sample_id = None):
  print(f'Getting first free index for {project_name}, parent {parent_sample_id}')
  child_ids = get_children(project_name, parent_sample_id, include_custom=False)
  print(f'Child ids: {child_ids}')
  # child_indices = [ int(child_id.split('-')[-1]) for child_id in child_ids ]
  child_indices = []
  for child_id in child_ids:
    suffix = child_id.split('-')[-1]
    # If not an integer, ignore
    if suffix.isdigit():
      child_indices += [ int(suffix) ]

  first_free_index = max(child_indices) + 1 if child_indices and max(child_indices) >= 0 else 1
  print(f'First free index: {first_free_index}')

  return first_free_index