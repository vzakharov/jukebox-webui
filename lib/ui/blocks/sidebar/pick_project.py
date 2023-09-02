from lib.navigation.get_project import get_project
import UI.project as UI
import UI.first
import UI.general
import UI.misc
import UI.navigation
import UI.main

def render_pick_project():
  UI.general.project_name.render().change(
    inputs = [ UI.general.project_name, UI.navigation.routed_sample_id ],
    outputs = [
      UI.general.create_project_box, UI.general.settings_box, *UI.project_settings, UI.misc.getting_started_column, UI.main.workspace_column, UI.first.first_generation_row,
      UI.navigation.sample_tree_row, UI.navigation.sample_box
    ],
    fn = get_project,
    api_name = 'get-project'
  )