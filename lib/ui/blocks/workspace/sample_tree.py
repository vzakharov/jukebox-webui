from lib.navigation.get_samples import get_samples

import gradio as gr

import lib.ui.components.general
import lib.ui.components.navigation

def render_sample_tree():
    with lib.ui.components.navigation.sample_tree_row.render():
      lib.ui.components.navigation.routed_sample_id.render()
      lib.ui.components.navigation.sample_tree.render()

      with gr.Column():
            # with gr.Accordion('Options & stats', open=False ):
        lib.ui.components.navigation.show_leafs_only.render()

        lib.ui.components.navigation.show_leafs_only.change(
              inputs = [ lib.ui.components.general.project_name, lib.ui.components.navigation.show_leafs_only ],
              outputs = lib.ui.components.navigation.sample_tree,
              fn = lambda *args: gr.update( choices = get_samples(*args) ),
            )