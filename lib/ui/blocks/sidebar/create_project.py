from lib.navigation.create_project import create_project
import lib.ui.UI as UI
from lib.utils import convert_name

import gradio as gr

def render_create_project():
  with UI.create_project_box.render():
    UI.new_project_name.render().blur(
      inputs = UI.new_project_name,
      outputs = UI.new_project_name,
      fn = convert_name,
    )

    # When a project is created, create a subfolder for it and update the project list.
    create_args = dict(
      inputs = UI.new_project_name,
      outputs = UI.project_name,
      fn = create_project,
    )

    UI.new_project_name.submit( **create_args )
    gr.Button('Create project').click( **create_args )