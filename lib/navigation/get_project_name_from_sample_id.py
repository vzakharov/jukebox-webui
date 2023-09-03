from .get_projects import get_projects


def get_project_name_from_sample_id(sample_id):
  projects = get_projects(include_new = False)
  # Find a project that matches the sample id, which is [project name]-[rest of sample id]
  for project_name in projects:
    if sample_id.startswith(f'{project_name}-'):
      return project_name