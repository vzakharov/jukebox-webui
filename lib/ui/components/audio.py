import gradio as gr


current_chunks = gr.File(
  elem_id = 'current-chunks',
  type = 'binary',
  visible = False,
  file_count = 'multiple'
)
sibling_chunks = gr.File(
  elem_id = 'sibling-chunks',
  type = 'binary',
  visible = False,
  file_count = 'multiple'
)
audio_waveform = gr.HTML(
  elem_id = 'audio-waveform'
)
audio_timeline = gr.HTML(
  elem_id = 'audio-timeline'
)