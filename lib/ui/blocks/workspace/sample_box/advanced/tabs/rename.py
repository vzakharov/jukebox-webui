from lib.navigation.rename_sample import rename_sample

import gradio as gr

from UI.general import project_name
from UI.navigation import picked_sample, show_leafs_only, sample_tree

def render_rename_tab():
  with gr.Tab('Rename sample'):
    new_sample_id = gr.Textbox(
      label = 'New sample id',
      placeholder = 'Alphanumeric and dashes only'
    )

    gr.Button('Rename').click(
      inputs = [ project_name, picked_sample, new_sample_id, show_leafs_only ],
      outputs = sample_tree,
      fn = rename_sample,
      api_name = 'rename-sample'
    )