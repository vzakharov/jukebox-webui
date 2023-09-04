### Meta

import gradio as gr

browser_timezone = gr.State(
  elem_id = 'browser-timezone',
)

separate_tab_warning = gr.Box(
  visible = False
)
separate_tab_link = gr.Textbox(
  visible = False
)
getting_started_column = gr.Column( scale = 2, elem_id = 'getting-started-column' )
generation_progress = gr.Markdown('Generation status will be shown here', elem_id = 'generation-progress')
