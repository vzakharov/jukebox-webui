from lib.model.generate import generate
import UI.misc as UI
import UI.first
import UI.general
import UI.navigation
import UI.project
from lib.ui.utils import HIDE, SHOW

import gradio as gr

def render_first_generation():
    with UI.first.first_generation_row.render():
      with gr.Column():
        gr.Markdown("""
              To start composing, you need to generate the first batch of samples. You can:
              - Start from scratch by clicking the **Generate initial samples** button below, or
              - Go to the **Prime** tab and convert your own audio to a sample.
            """)

        gr.Button('Generate initial samples', variant = "primary" ).click(
              inputs = [ UI.general.project_name, UI.navigation.sample_tree, UI.navigation.show_leafs_only, *UI.project.generation_params ],
              outputs = [ UI.navigation.sample_tree, UI.first.first_generation_row, UI.navigation.sample_tree_row, UI.generation_progress ],
              fn = lambda *args: {
                **generate(*args),
                UI.first.first_generation_row: HIDE,
                UI.navigation.sample_tree_row: SHOW,
              }
            )