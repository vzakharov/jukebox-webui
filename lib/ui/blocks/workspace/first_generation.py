from lib.model.generate import generate
import UI.misc as UI
from UI.first import first_generation_row, first_generation_row, first_generation_row
from UI.general import project_name
from UI.navigation import sample_tree, show_leafs_only, sample_tree, sample_tree_row, sample_tree_row
from UI.project import generation_params
from lib.ui.utils import HIDE, SHOW

import gradio as gr

def render_first_generation():
    with first_generation_row.render():
      with gr.Column():
        gr.Markdown("""
              To start composing, you need to generate the first batch of samples. You can:
              - Start from scratch by clicking the **Generate initial samples** button below, or
              - Go to the **Prime** tab and convert your own audio to a sample.
            """)

        gr.Button('Generate initial samples', variant = "primary" ).click(
              inputs = [ project_name, sample_tree, show_leafs_only, *generation_params ],
              outputs = [ sample_tree, first_generation_row, sample_tree_row, UI.generation_progress ],
              fn = lambda *args: {
                **generate(*args),
                first_generation_row: HIDE,
                sample_tree_row: SHOW,
              }
            )