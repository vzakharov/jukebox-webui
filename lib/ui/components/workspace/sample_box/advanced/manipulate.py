from lib.cut import cut_audio
from lib.ui.UI import UI
from lib.ui.preview import default_preview_args

import gradio as gr

import subprocess

def render_manipulate(app):
    with gr.Tab('Manipulate audio'):
      UI.total_audio_length.render()

          # Change the max n samples depending on the audio length
      def set_max_n_samples( total_audio_length, n_samples ):
        max_n_samples_by_gpu_and_duration = {
              'Tesla T4': {
                0: 4,
                8.5: 3,
                13: 2
              }
              # The key indicates the audio length threshold in seconds trespassing which makes max_n_samples equal to the value
            }

            # Get GPU via nvidia-smi
        gpu = subprocess.check_output( 'nvidia-smi --query-gpu=gpu_name --format=csv,noheader', shell=True ).decode('utf-8').strip()

            # The default value is 4
        max_n_samples = 4
        if gpu in max_n_samples_by_gpu_and_duration and total_audio_length:
              # Get the max n samples for the GPU from the respective dict
          max_n_samples_for_gpu = max_n_samples_by_gpu_and_duration[gpu]
          max_n_samples_above_threshold = [ max_n_samples_for_gpu[threshold] for threshold in max_n_samples_for_gpu if total_audio_length > threshold ]
          if len(max_n_samples_above_threshold) > 0:
            max_n_samples = min( max_n_samples_for_gpu[ threshold ] for threshold in max_n_samples_for_gpu if total_audio_length > threshold )

        return max_n_samples

          # Do this on audio length change and app load
      for handler in [ UI.total_audio_length.change, app.load ]:
        handler(
              inputs = [ UI.total_audio_length, UI.n_samples ],
              outputs = UI.max_n_samples,
              fn = set_max_n_samples,
            )

          # If max_n_samples changes, update the n_samples input respectively
      UI.max_n_samples.render().change(
            inputs = UI.max_n_samples,
            outputs = UI.n_samples,
            fn = lambda max_n_samples: gr.update(
              maximum = max_n_samples,
              # value = min( n_samples, max_n_samples ),
            )
          )

      UI.cut_audio_specs.render().submit(**default_preview_args)

      with gr.Row():
        UI.cut_audio_preview_button.render().click(**default_preview_args)

            # Make the cut out buttons visible or not depending on whether the cut out value is 0
        UI.cut_audio_specs.change(
              inputs = UI.cut_audio_specs,
              outputs = [ UI.cut_audio_preview_button, UI.cut_audio_apply_button ],
              fn = lambda cut_audio_specs: [
                gr.update( visible = cut_audio_specs != '' ) for _ in range(3)
              ]
            )

        UI.cut_audio_apply_button.render().click(
              inputs = [ UI.project_name, UI.picked_sample, UI.cut_audio_specs ],
              outputs = UI.cut_audio_specs,
              fn = cut_audio,
              api_name = 'cut-audio',
            )

      with gr.Accordion('How does it work?', open = False):
            # possible specs:
            # start-end -- cuts out the specified range (in seconds), either can be omitted to cut from the start or to the end, the dash can be omitted to cut from the specified time to the end
            # start-end+sample_id@start-end -- takes the specified range from the specified sample and adds it instead of the cut-out range
            # start-end+start-end -- same, but takes the specified range from the current sample
            # +sample_id@start-end -- adds the specified range from the specified sample to the end of the current sample
            # +start-end -- keeps just the specified range from the current sample (i.e. the opposite of start-end)
            # Any whitespaces are ignored
        gr.Markdown('''
              - `start-end` (e.g. 0.5-2.5) — *removes* the specified range (in seconds),
                - `start-` or just `start` — *removes* from the specified time to the end
                - `-end` -- **removes** from the start to the specified time
              - `start-end+start-end` — *removes* the range before `+` and *inserts* the range after `+` instead. Note that, unlike the remove range, the insert range must be fully specified.
              - `start-end+sample_id@start-end` — same as above, but the insert range is taken from the specified sample, even if it is in another project (mix and match!)
              - `+sample_id@start-end` — same as above, but the range from the other sample is added *to the end* of the current sample
              - `+start-end` — *keeps* just the specified range and removes everything else.
              You can combine several of the above by using a comma (`,`). **KEEP IN MIND** that in this case the ranges are applied sequentially, so the order matters. For example, `0-1,2-3` will first remove 0-1s, and will then remove 2-3s FROM THE ALREADY MODIFIED SAMPLE, so it will actually remove ranges 0-1s and *3-4s* from the original sample. This is intentional, as it allows for a step-by-step approach to editing the audio, where you add new specifiers as you listen to the result of the previous ones.
            ''')

      UI.preview_just_the_last_n_sec.render().blur(**default_preview_args)