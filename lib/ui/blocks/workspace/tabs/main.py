import gradio as gr

from lib.app import app
from lib.navigation.get_sample_filename import get_sample_filename
from lib.navigation.get_sibling_samples import get_sibling_samples
from lib.navigation.refresh_siblings import refresh_siblings
from lib.ui.blocks.workspace.first_generation import \
    render_first_generation
from lib.ui.blocks.workspace.sample_box.sample_box import render_sample_box
from lib.ui.blocks.workspace.sample_tree import render_sample_tree
from UI.audio import sibling_chunks, current_chunks, sibling_chunks
from UI.general import project_name
from UI.misc import generation_progress
from UI.navigation import picked_sample, sample_tree, sample_tree, picked_sample, picked_sample, picked_sample_updated
from UI.upsampling import upsampled_lengths, upsampled_lengths
from lib.ui.js.update_url import update_url_js
from lib.ui.preview import default_preview_args, preview_inputs

def render_main_workspace_tab():
    with gr.Tab('Workspace'):
      with gr.Column():
        render_first_generation()
        render_sample_tree()

        picked_sample.render()

        sample_tree.change(
          inputs = [ project_name, sample_tree ],
          outputs = picked_sample,
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

        picked_sample.change(
          **default_preview_args,
          api_name = 'get-sample',
          _js = update_url_js
        )

        # When the picked sample is updated, update all the others too (UI.sibling_chunks) by calling get_sample for each sibling
        picked_sample_updated.render().change(
          inputs = [ *preview_inputs ],
          outputs = sibling_chunks,
          fn = get_sibling_samples,
          api_name = 'get-sibling-samples',
        )

        current_chunks.render()
        sibling_chunks.render()

        upsampled_lengths.render().change(
          inputs = upsampled_lengths,
          outputs = None,
          fn = None,
          # Split by comma and turn into floats and add wavesurfer markers for each (first clear all the markers)
          _js = 'comma_separated => Ji.addUpsamplingMarkers( comma_separated.split(",").map( parseFloat ) )'
        )

        render_sample_box()

      generation_progress.render()