from lib.navigation.get_project import get_project
import lib.ui.UI as UI

def render_pick_project():
  UI.project_name.render().change(
    inputs = [ UI.project_name, UI.routed_sample_id ],
    outputs = [
      UI.create_project_box, UI.settings_box, *UI.project_settings, UI.getting_started_column, UI.workspace_column, UI.first_generation_row,
      UI.sample_tree_row, UI.sample_box
    ],
    fn = get_project,
    api_name = 'get-project'
  )