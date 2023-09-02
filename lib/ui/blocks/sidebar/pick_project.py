from lib.navigation.get_project import get_project
from UI.project import project_settings
from UI.first import first_generation_row
from UI.general import project_name, project_name, create_project_box, settings_box
from UI.misc import getting_started_column
from UI.navigation import routed_sample_id, sample_tree_row
from UI.sample import sample_box
from UI.main import workspace_column

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