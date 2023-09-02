from lib.navigation.rename_sample import rename_sample
import lib.ui.components as UI

import gradio as gr

def render_rename_tab():
  with gr.Tab('Rename sample'):
    new_sample_id = gr.Textbox(
      label = 'New sample id',
      placeholder = 'Alphanumeric and dashes only'
    )

    gr.Button('Rename').click(
      inputs = [ UI.project_name, UI.picked_sample, new_sample_id, UI.show_leafs_only ],
      outputs = UI.sample_tree,
      fn = rename_sample,
      api_name = 'rename-sample'
    )