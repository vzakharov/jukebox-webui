import gradio as gr


sample_box = gr.Box(
  visible = False
)
compose_row = gr.Box(
  elem_id = 'compose-row',
)
go_to_parent_button = gr.Button(
  value = '<',
)
go_to_children_button = gr.Button(
  value = '>',
)