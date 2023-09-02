import re

from lib.navigation.get_last_project import get_last_project


def route_from_url(query_string, projects):
    if query_string:
      print(f'Query string: {query_string}')
      if '-' in query_string:
        project_name, sample_id = re.match('^(.*?)-(.*)$', query_string).groups()
        sample_id = f'{project_name}-{sample_id}'
        print(f'Routed to project {project_name} and sample {sample_id}')
      else:
        project_name = query_string
        sample_id = None
        print(f'Routed to project {project_name}')
    else:
      project_name = get_last_project(projects)
      sample_id = None
    return project_name,sample_id