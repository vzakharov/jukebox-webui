from lib.navigation.rename_sample import rename_sample

import gradio as gr

import lib.ui.components.general
import lib.ui.components.navigation

def render_rename_tab():
  with gr.Tab('Rename sample'):
    new_sample_id = gr.Textbox(
      label = 'New sample id',
      placeholder = 'Alphanumeric and dashes only'
    )

    gr.Button('Rename').click(
      inputs = [ lib.ui.components.general.project_name, lib.ui.components.navigation.picked_sample, new_sample_id, lib.ui.components.navigation.show_leafs_only ],
      outputs = lib.ui.components.navigation.sample_tree,
      fn = rename_sample,
      api_name = 'rename-sample'
    )