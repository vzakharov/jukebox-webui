import gradio as gr


from datetime import datetime


class UI:

  ### Meta

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

  ### General

  project_name = gr.Dropdown(
    label = 'Project'
  )

  create_project_box = gr.Box(
    visible = False
  )

  new_project_name = gr.Textbox(
    label = 'Project name',
    placeholder = 'lowercase-digits-and-dashes-only'
  )

  settings_box = gr.Accordion(
    label = "Settings",
    visible = False
  )

  general_settings = [ project_name ]

  ### Project-specific

  ## Metas (artist, genre, lyrics)
  artist = gr.Dropdown(
    label = 'Artist'
  )

  genre_dropdown = gr.Dropdown(
    label = 'Available genres'
  )

  genre = gr.Textbox(
    label = 'Genre',
    placeholder = 'Separate several with spaces'
  )

  lyrics = gr.Textbox(
    label = 'Lyrics (optional)',
    max_lines = 5,
    placeholder = 'Shift+Enter for a new line'
  )

  metas = [ artist, genre, lyrics ]

  n_samples = gr.Slider(
    label = 'Number of samples',
    minimum = 1,
    maximum = 5,
    step = 1
  )

  max_n_samples = gr.Number(
    visible = False
  )

  temperature = gr.Slider(
    label = 'Temperature',
    minimum = 0.9,
    maximum = 1.1,
    step = 0.005
  )

  generation_length = gr.Slider(
    label = 'Generation length, sec',
    minimum = 0.5,
    maximum = 10,
    step = 0.1
  )

  generation_discard_window = gr.Slider(
    label = 'Generation discard window, sec',
    minimum = 0,
    maximum = 200,
    step = 1
  )

  generation_params = [ artist, genre, lyrics, n_samples, temperature, generation_length, generation_discard_window ]

  getting_started_column = gr.Column( scale = 2, elem_id = 'getting-started-column' )

  workspace_column = gr.Column( scale = 3, visible = False )

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

  generation_progress = gr.Markdown('Generation status will be shown here', elem_id = 'generation-progress')

  routed_sample_id = gr.State()

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

  sample_box = gr.Box(
    visible = False
  )

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

  upsampled_lengths = gr.Textbox(visible = False)
  # (Comma-separated list of audio lengths by upsampling level, e.g. '0.5,1'. If only midsampled audio is available, the list will only contain one element, e.g. '1'.)

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

  compose_row = gr.Box(
    elem_id = 'compose-row',
  )

  go_to_parent_button = gr.Button(
    value = '<',
  )

  go_to_children_button = gr.Button(
    value = '>',
  )

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

  upsampling_audio_refresher = gr.Number( value = 0, visible = False )
  # Note: for some reason, Gradio doesn't monitor programmatic changes to a checkbox, so we use a number instead

  upsampling_refresher = gr.Number( value = 0, visible = False )

  upsampling_running = gr.Number( visible = False )

  upsampling_triggered_by_button = gr.Checkbox( visible = False, value = False )

  project_settings = [
    *generation_params, sample_tree, show_leafs_only, preview_just_the_last_n_sec,
    genre_for_upsampling_left_channel, genre_for_upsampling_center_channel, genre_for_upsampling_right_channel
  ]

  input_names = { input: name for name, input in locals().items() if isinstance(input, gr.components.FormComponent) }

  inputs_by_name = { name: input for name, input in locals().items() if isinstance(input, gr.components.FormComponent) }