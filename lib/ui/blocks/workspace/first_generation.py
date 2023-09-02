from lib.model.generate import generate
import lib.ui.components as UI
from lib.ui.utils import HIDE, SHOW

import gradio as gr

def render_first_generation():
    with UI.first_generation_row.render():
      with gr.Column():
        gr.Markdown("""
              To start composing, you need to generate the first batch of samples. You can:
              - Start from scratch by clicking the **Generate initial samples** button below, or
              - Go to the **Prime** tab and convert your own audio to a sample.
            """)

        gr.Button('Generate initial samples', variant = "primary" ).click(
              inputs = [ UI.project_name, UI.sample_tree, UI.show_leafs_only, *UI.generation_params ],
              outputs = [ UI.sample_tree, UI.first_generation_row, UI.sample_tree_row, UI.generation_progress ],
              fn = lambda *args: {
                **generate(*args),
                UI.first_generation_row: HIDE,
                UI.sample_tree_row: SHOW,
              }
            )