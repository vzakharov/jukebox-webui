import gradio as gr


upsampling_accordion = gr.Accordion(
  label = 'Upsampling',
  open = False
  # visible = False
)
UPSAMPLING_LEVEL_NAMES = [ 'Raw', 'Midsampled', 'Upsampled' ]
upsampling_level = gr.Dropdown(
  label = 'Upsampling level',
  choices = [ 'Raw' ],
  value = 'Raw',
)
upsample_rendering = gr.Dropdown(
  label = 'Render...',
  type = 'index',
  choices = [ 'Channel 1', 'Channel 2', 'Channel 3', 'Pseudo-stereo', 'Pseudo-stereo with delay' ],
  value = 'Pseudo-stereo',
)
combine_upsampling_levels = gr.Checkbox(
  label = 'Combine levels',
  value = True
)
invert_center_channel = gr.Checkbox(
  label = 'Invert center channel',
  value = False
)
continue_upsampling_button = gr.Button('Continue upsampling', visible = False )
# (Comma-separated list of audio lengths by upsampling level, e.g. '0.5,1'. If only midsampled audio is available, the list will only contain one element, e.g. '1'.)
upsampled_lengths = gr.Textbox(visible = False)
sample_to_upsample = gr.Textbox(
  label = 'Sample to upsample',
  placeholder = 'Choose a sample in the Workspace tab first',
  interactive = False,
)
genre_for_upsampling_left_channel = gr.Dropdown(
  label = 'Genre for upsampling (left channel)'
)
genre_for_upsampling_center_channel = gr.Dropdown(
  label = 'Genre for upsampling (center channel)'
)
genre_for_upsampling_right_channel = gr.Dropdown(
  label = 'Genre for upsampling (right channel)'
)
kill_runtime_once_done = gr.Checkbox(
  label = 'Kill runtime once done',
  value = False
)
upsample_button = gr.Button('Start upsampling', variant="primary", elem_id='upsample-button')
upsampling_status = gr.Markdown('Upsampling progress will be shown here', visible = False)
# Note: for some reason, Gradio doesn't monitor programmatic changes to a checkbox, so we use a number instead
upsampling_audio_refresher = gr.Number( value = 0, visible = False )
upsampling_refresher = gr.Number( value = 0, visible = False )
upsampling_running = gr.Number( visible = False )
upsampling_triggered_by_button = gr.Checkbox( visible = False, value = False )