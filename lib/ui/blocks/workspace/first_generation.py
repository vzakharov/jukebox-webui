from lib.model.generate import generate
import lib.ui.components.misc as UI
import lib.ui.components.first
import lib.ui.components.general
import lib.ui.components.navigation
import lib.ui.components.project
from lib.ui.utils import HIDE, SHOW

import gradio as gr

def render_first_generation():
    with lib.ui.components.first.first_generation_row.render():
      with gr.Column():
        gr.Markdown("""
              To start composing, you need to generate the first batch of samples. You can:
              - Start from scratch by clicking the **Generate initial samples** button below, or
              - Go to the **Prime** tab and convert your own audio to a sample.
            """)

        gr.Button('Generate initial samples', variant = "primary" ).click(
              inputs = [ lib.ui.components.general.project_name, lib.ui.components.navigation.sample_tree, lib.ui.components.navigation.show_leafs_only, *lib.ui.components.project.generation_params ],
              outputs = [ lib.ui.components.navigation.sample_tree, lib.ui.components.first.first_generation_row, lib.ui.components.navigation.sample_tree_row, UI.generation_progress ],
              fn = lambda *args: {
                **generate(*args),
                lib.ui.components.first.first_generation_row: HIDE,
                lib.ui.components.navigation.sample_tree_row: SHOW,
              }
            )