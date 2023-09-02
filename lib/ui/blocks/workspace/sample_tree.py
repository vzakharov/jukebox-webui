from lib.navigation.get_samples import get_samples
import lib.ui.UI as UI

import gradio as gr

def render_sample_tree():
    with UI.sample_tree_row.render():
      UI.routed_sample_id.render()
      UI.sample_tree.render()

      with gr.Column():
            # with gr.Accordion('Options & stats', open=False ):
        UI.show_leafs_only.render()

        UI.show_leafs_only.change(
              inputs = [ UI.project_name, UI.show_leafs_only ],
              outputs = UI.sample_tree,
              fn = lambda *args: gr.update( choices = get_samples(*args) ),
            )