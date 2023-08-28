from lib.navigation.get_projects import get_projects


import gradio as gr


def set_get_projects_api():
  project_list = gr.CheckboxGroup(
    visible = False,
  )

  project_list.change(
    inputs = None,
    outputs = [ project_list ],
    fn = lambda: get_projects(),
    api_name = 'get-projects'
  )