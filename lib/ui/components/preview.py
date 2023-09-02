import gradio as gr


total_audio_length = gr.Number(
  label = 'Total audio length, sec',
  elem_id = 'total-audio-length',
  interactive = False,
  visible = False
)
preview_just_the_last_n_sec = gr.Number(
  label = 'Preview the last ... seconds',
  elem_id = 'preview-last-n-sec'
)
cut_audio_specs = gr.Textbox(
  label = 'Cut, trim, merge',
  placeholder = 'See accordion below for syntax',
  elem_id = 'cut-audio-specs',
)
cut_audio_preview_button = gr.Button( 'Preview', visible = False, variant = 'secondary' )
cut_audio_apply_button = gr.Button( 'Apply', visible = False, variant = 'primary' )