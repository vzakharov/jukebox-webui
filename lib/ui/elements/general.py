### General

import gradio as gr

project_name = gr.Dropdown(
  label = 'Project'
)
create_project_box = gr.Box(
  visible = False
)
new_project_name = gr.Textbox(
  label = 'Project name',
  placeholder = 'lowercase-digits-and-dashes-only'
)
settings_box = gr.Accordion(
  label = "Settings",
  visible = False
)
general_settings = [ project_name ]