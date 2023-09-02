from lib.navigation.rename_sample import rename_sample

import gradio as gr

import UI.general
import UI.navigation

def render_rename_tab():
  with gr.Tab('Rename sample'):
    new_sample_id = gr.Textbox(
      label = 'New sample id',
      placeholder = 'Alphanumeric and dashes only'
    )

    gr.Button('Rename').click(
      inputs = [ UI.general.project_name, UI.navigation.picked_sample, new_sample_id, UI.navigation.show_leafs_only ],
      outputs = UI.navigation.sample_tree,
      fn = rename_sample,
      api_name = 'rename-sample'
    )