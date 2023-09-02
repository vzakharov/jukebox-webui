from lib.navigation.get_samples import get_samples

import gradio as gr

from UI.general import project_name
from UI.navigation import sample_tree_row, routed_sample_id, sample_tree, show_leafs_only, show_leafs_only, show_leafs_only, sample_tree

def render_sample_tree():
    with sample_tree_row.render():
      routed_sample_id.render()
      sample_tree.render()

      with gr.Column():
            # with gr.Accordion('Options & stats', open=False ):
        show_leafs_only.render()

        show_leafs_only.change(
              inputs = [ project_name, show_leafs_only ],
              outputs = sample_tree,
              fn = lambda *args: gr.update( choices = get_samples(*args) ),
            )