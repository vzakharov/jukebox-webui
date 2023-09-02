### Meta

import gradio as gr

browser_timezone = gr.State()

separate_tab_warning = gr.Box(
  visible = False
)
separate_tab_link = gr.Textbox(
  visible = False
)
main_window = gr.Row(
  visible = False
)