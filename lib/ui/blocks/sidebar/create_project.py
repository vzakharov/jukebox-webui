from lib.navigation.create_project import create_project
from lib.ui.elements.general import create_project_box, new_project_name, new_project_name, new_project_name, new_project_name, project_name, new_project_name
from lib.utils import convert_name

import gradio as gr

def render_create_project():
  with create_project_box.render():
    new_project_name.render().blur(
      inputs = new_project_name,
      outputs = new_project_name,
      fn = convert_name,
    )

    # When a project is created, create a subfolder for it and update the project list.
    create_args = dict(
      inputs = new_project_name,
      outputs = project_name,
      fn = create_project,
    )

    new_project_name.submit( **create_args )
    gr.Button('Create project').click( **create_args )