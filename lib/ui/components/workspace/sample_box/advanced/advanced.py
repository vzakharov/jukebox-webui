from lib.ui.components.workspace.sample_box.advanced.tabs.completify import render_completify_tab
from lib.ui.components.workspace.sample_box.advanced.tabs.purge import render_purge_tab
from lib.ui.components.workspace.sample_box.advanced.tabs.rename import render_rename_tab
from lib.ui.components.workspace.sample_box.advanced.tabs.manipulate import render_manipulate_tab

import gradio as gr

def render_advanced(app):
  with gr.Accordion( 'Advanced', open = False ):    
    render_manipulate_tab(app)

    render_rename_tab()

    render_purge_tab()

    render_completify_tab()

