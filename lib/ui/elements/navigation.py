import gradio as gr

routed_sample_id = gr.State(
  elem_id = 'routed-sample-id',
)
sample_tree_row = gr.Row(
  visible = False
)
sample_tree = gr.Dropdown(
  label = 'Sample tree',
)
show_leafs_only = gr.Checkbox(
  label = 'Leaf samples only',
)
branch_sample_count = gr.Number(
  label = '# branch samples',
)
leaf_sample_count = gr.Number(
  label = '# leaf samples',
)
picked_sample = gr.Radio(
  label = 'Variations',
)
picked_sample_updated = gr.Number( 0, visible = False )
