from lib.navigation.get_project import get_project
import lib.ui.components.project as UI
import lib.ui.components.first
import lib.ui.components.general
import lib.ui.components.misc
import lib.ui.components.navigation
import lib.ui.components.main

def render_pick_project():
  lib.ui.components.general.project_name.render().change(
    inputs = [ lib.ui.components.general.project_name, lib.ui.components.navigation.routed_sample_id ],
    outputs = [
      lib.ui.components.general.create_project_box, lib.ui.components.general.settings_box, *UI.project_settings, lib.ui.components.misc.getting_started_column, lib.ui.components.main.workspace_column, lib.ui.components.first.first_generation_row,
      lib.ui.components.navigation.sample_tree_row, lib.ui.components.navigation.sample_box
    ],
    fn = get_project,
    api_name = 'get-project'
  )