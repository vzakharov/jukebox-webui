from lib.navigation.get_samples import get_samples

import gradio as gr

import UI.general
import UI.navigation

def render_sample_tree():
    with UI.navigation.sample_tree_row.render():
      UI.navigation.routed_sample_id.render()
      UI.navigation.sample_tree.render()

      with gr.Column():
            # with gr.Accordion('Options & stats', open=False ):
        UI.navigation.show_leafs_only.render()

        UI.navigation.show_leafs_only.change(
              inputs = [ UI.general.project_name, UI.navigation.show_leafs_only ],
              outputs = UI.navigation.sample_tree,
              fn = lambda *args: gr.update( choices = get_samples(*args) ),
            )