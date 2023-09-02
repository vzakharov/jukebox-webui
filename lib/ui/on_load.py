
import gradio as gr

from lib.navigation.get_projects import get_projects

from .elements.general import project_name
from .elements.main import main_window
from .elements.misc import separate_tab_link, separate_tab_warning
from .initial_states import initial_states
from .route_from_url import route_from_url


def on_load( href, query_string, error_message ):

  if error_message:
    print(f'Please open this app in a separate browser tab: {href}')
    print(f'Error message from the client (for debugging only; you can ignore this): {error_message}')

    return {
      separate_tab_warning: gr.update(
        visible = True,
      ),
      separate_tab_link: href,
      main_window: gr.update(
        visible = False,
      ),
    }

  projects = get_projects()

  # If there is a query string, it will be of the form project_name-sample_id or project_name
  project_name, sample_id = route_from_url(query_string, projects)

  return initial_states(projects, project_name, sample_id)