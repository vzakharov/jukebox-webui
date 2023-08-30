import gradio as gr

from lib.app import app
from lib.navigation.get_sample_filename import get_sample_filename
from lib.navigation.get_sibling_samples import get_sibling_samples
from lib.navigation.refresh_siblings import refresh_siblings
from lib.ui.components.workspace.first_generation import \
    render_first_generation
from lib.ui.components.workspace.sample_box.sample_box import render_sample_box
from lib.ui.components.workspace.sample_tree import render_sample_tree
from lib.ui.js.update_url import update_url_js
from lib.ui.preview import default_preview_args, preview_inputs
from lib.ui.UI import UI


def render_main_workspace_tab():
    with gr.Tab('Workspace'):
      with gr.Column():
        render_first_generation()
        render_sample_tree()

        UI.picked_sample.render()

        UI.sample_tree.change(
          inputs = [ UI.project_name, UI.sample_tree ],
          outputs = UI.picked_sample,
          fn = refresh_siblings,
          api_name = 'get-siblings'
        )

        # Virtual input & handler to create an API method for get_sample_filename
        gr.Textbox(visible=False).change(
          inputs = preview_inputs,
          outputs = gr.Textbox(visible=False),
          fn = get_sample_filename,
          api_name = 'get-sample-filename'
        )

        UI.picked_sample.change(
          **default_preview_args,
          api_name = 'get-sample',
          _js = update_url_js
        )

        # When the picked sample is updated, update all the others too (UI.sibling_chunks) by calling get_sample for each sibling
        UI.picked_sample_updated.render().change(
          inputs = [ *preview_inputs ],
          outputs = UI.sibling_chunks,
          fn = get_sibling_samples,
          api_name = 'get-sibling-samples',
        )

        UI.current_chunks.render()
        UI.sibling_chunks.render()

        UI.upsampled_lengths.render().change(
          inputs = UI.upsampled_lengths,
          outputs = None,
          fn = None,
          # Split by comma and turn into floats and add wavesurfer markers for each (first clear all the markers)
          _js = 'comma_separated => Ji.addUpsamplingMarkers( comma_separated.split(",").map( parseFloat ) )'
        )

        render_sample_box()

      UI.generation_progress.render()