from lib.navigation.get_project import get_project
from lib.ui.elements.project import project_settings
from lib.ui.elements.first import first_generation_row
from lib.ui.elements.general import project_name, project_name, create_project_box, settings_box
from lib.ui.elements.misc import getting_started_column
from lib.ui.elements.navigation import routed_sample_id, sample_tree_row
from lib.ui.elements.sample import sample_box
from lib.ui.elements.main import workspace_column

def render_pick_project():
  project_name.render().change(
    inputs = [ project_name, routed_sample_id ],
    outputs = [
      create_project_box, settings_box, *project_settings, getting_started_column, workspace_column, first_generation_row,
      sample_tree_row, sample_box
    ],
    fn = get_project,
    api_name = 'get-project'
  )