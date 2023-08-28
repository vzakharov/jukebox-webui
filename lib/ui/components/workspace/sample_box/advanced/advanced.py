from lib.navigation.purging import purge_samples
from lib.navigation.rename_sample import rename_sample
from lib.navigation.utils import backup_sample, get_zs, save_zs
from lib.ui.UI import UI
from lib.ui.components.workspace.sample_box.advanced.manipulate import render_manipulate
from lib.navigation.purging import prepare_purge_list

import gradio as gr

def render_advanced(app):
  with gr.Accordion( 'Advanced', open = False ):    
    render_manipulate(app)

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

    with gr.Tab('Completify sample'):
      gr.Markdown('''
            For space saving purposes, the app will sometime NOT include the entire information needed to render the sample into the sample file, taking the missing info (e.g. upsampled tokens) from its ancestors instead.
            If, for whatever reason, you want to have the entire information in the sample file, you can add it by clicking the button below.
          ''')

      def completify(project_name, sample_id):
        zs = get_zs(project_name, sample_id, True)
        backup_sample(project_name, sample_id)
        save_zs(zs, project_name, sample_id)

      completify_button = gr.Button('Completify')
      completify_button.click(
            completify,
            [ UI.project_name, UI.picked_sample ],
            gr.Button('Completify')
          )

