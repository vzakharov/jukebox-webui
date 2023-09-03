from datetime import datetime
import gradio as gr

primed_audio = gr.Audio(
  label = 'Audio to start from (optional)',
  source = 'microphone'
)
# Virtual timestamp textbox to do certain things once the audio is primed (and this textbox is updated), accurate to the millisecond
prime_timestamp = gr.Textbox(
  value = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
  visible = False
)
first_generation_row = gr.Row(
  visible = False
)