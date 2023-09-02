from lib.navigation.purging import prepare_purge_list, purge_samples
import lib.ui.components.general as UI

import gradio as gr

def render_purge_tab():
  with gr.Tab('Purge samples'):
    # For all samples whose parent sample's level 0/1 are the same as this one, purge those levels
    # We need a button to prepare the list of samples to purge, a multiline textbox to show the list, and a button to confirm the purge
    purge_list = gr.Textbox(
      label = 'Purge list',
      placeholder = 'Click the button below to prepare the list of samples to purge',
      multiline = True,
      disabled = True,
    )

    gr.Button('Prepare purge list').click(
      inputs = [ UI.project_name ],
      outputs = purge_list,
      fn = prepare_purge_list,
      api_name = 'prepare-purge-list'
    )

    gr.Button('Purge samples').click(
      inputs = [ UI.project_name, purge_list ],
      outputs = purge_list,
      fn = purge_samples,
      api_name = 'purge-samples'
    )