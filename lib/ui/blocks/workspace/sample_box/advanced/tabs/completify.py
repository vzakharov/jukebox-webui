from lib.navigation.utils import backup_sample, get_zs, save_zs
from lib.ui.elements.navigation import picked_sample

import gradio as gr

from lib.ui.elements.general import project_name

def completify(project_name, sample_id):
  zs = get_zs(project_name, sample_id, True)
  backup_sample(project_name, sample_id)
  save_zs(zs, project_name, sample_id)

def render_completify_tab():
  with gr.Tab('Completify sample'):
    gr.Markdown('''
      For space saving purposes, the app will sometime NOT include the entire information needed to render the sample into the sample file, taking the missing info (e.g. upsampled tokens) from its ancestors instead.
      If, for whatever reason, you want to have the entire information in the sample file, you can add it by clicking the button below.
    ''')

    completify_button = gr.Button('Completify')
    completify_button.click(
      completify,
      [ project_name, picked_sample ],
      gr.Button('Completify')
    )