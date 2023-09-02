from lib.navigation.create_project import create_project
import lib.ui.components.general
from lib.utils import convert_name

import gradio as gr

def render_create_project():
  with lib.ui.components.general.create_project_box.render():
    lib.ui.components.general.new_project_name.render().blur(
      inputs = lib.ui.components.general.new_project_name,
      outputs = lib.ui.components.general.new_project_name,
      fn = convert_name,
    )

    # When a project is created, create a subfolder for it and update the project list.
    create_args = dict(
      inputs = lib.ui.components.general.new_project_name,
      outputs = lib.ui.components.general.project_name,
      fn = create_project,
    )

    lib.ui.components.general.new_project_name.submit( **create_args )
    gr.Button('Create project').click( **create_args )