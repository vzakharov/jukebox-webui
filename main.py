import hashlib
import os
import random
import subprocess
import sys
import urllib.request
from datetime import timedelta, timezone

import gradio as gr
import torch as t
from jukebox.sample import load_prompts
from jukebox.utils.torch_utils import empty_cache
from lib.ui.utils import SHOW

from lib.api import set_get_projects_api
from lib.audio.convert_audio_to_sample import convert_audio_to_sample
from lib.cut import cut_audio, cut_zs
from lib.generate import generate
from lib.model.load import load_model
from lib.model.params import hps, set_hyperparams
from lib.navigation.delete_sample import delete_sample
from lib.navigation.get_children import get_children
from lib.navigation.get_parent import get_parent
from lib.navigation.get_sample import get_sample
from lib.navigation.get_sample_filename import get_sample_filename
from lib.navigation.get_samples import get_samples
from lib.navigation.get_sibling_samples import get_sibling_samples
from lib.navigation.refresh_siblings import refresh_siblings
from lib.navigation.rename_sample import rename_sample
from lib.navigation.utils import (backup_sample, get_zs, get_zs_filename,
                                  save_zs)
from lib.ui.app_layout import app_layout
from lib.ui.on_load import on_load
from lib.ui.UI import UI
from lib.ui.utils import HIDE
from lib.upsampling.start_upsampling import start_upsampling
from lib.upsampling.Upsampling import Upsampling
from lib.upsampling.utils import get_levels
from lib.utils import tokens_to_seconds
from params import DEV_MODE, GITHUB_SHA, base_path, debug_gradio, share_gradio
from lib.ui.sidebar.render import render_sidebar

print("Launch arguments:", sys.argv)

if '--no-load' in sys.argv:
  print("🚫 Skipping model loading")
  pass

else:

  device, browser_timezone, keep_upsampling_after_restart, vqvae, priors, top_prior = load_model(hps)

# If the base folder doesn't exist, create it
if not os.path.isdir(base_path):
  os.makedirs(base_path)

try:
  calculated_metas
  print('Calculated metas already loaded.')
except:
  calculated_metas = {}
  print('Calculated metas created.')

with app_layout() as app:

  UI.browser_timezone.render()

  # Render an invisible checkbox group to enable loading list of projects via API
  set_get_projects_api()

  with UI.separate_tab_warning.render():

    UI.separate_tab_link.render()

    gr.Button('Click here to open the UI', variant = 'primary' ).click( inputs = UI.separate_tab_link, outputs = None, fn = None,
      _js = "link => window.open(link, '_blank')"
    )
  
  with UI.main_window.render():

    render_sidebar()

    with UI.getting_started_column.render():

      # Load the getting started text from github (vzakharov/jukebox-webui/docs/getting-started.md) via urllib
      with urllib.request.urlopen('https://raw.githubusercontent.com/vzakharov/jukebox-webui/main/docs/getting-started.md') as f:
        getting_started_text = f.read().decode('utf-8')
        gr.Markdown(getting_started_text)

    with UI.workspace_column.render():

      with gr.Tab('Workspace'):

        with gr.Column():

          with UI.first_generation_row.render():

            with gr.Column():
            
              gr.Markdown("""
                To start composing, you need to generate the first batch of samples. You can:
                
                - Start from scratch by clicking the **Generate initial samples** button below, or
                - Go to the **Prime** tab and convert your own audio to a sample.
              """)

              gr.Button('Generate initial samples', variant = "primary" ).click(
                inputs = [ UI.project_name, UI.sample_tree, UI.show_leafs_only, *UI.generation_params ],
                outputs = [ UI.sample_tree, UI.first_generation_row, UI.sample_tree_row, UI.generation_progress ],
                fn = lambda *args: {
                  **generate(*args),
                  UI.first_generation_row: HIDE,
                  UI.sample_tree_row: SHOW,
                }
              )

          with UI.sample_tree_row.render():
            
            UI.routed_sample_id.render()
            UI.sample_tree.render()

            with gr.Column():

              # with gr.Accordion('Options & stats', open=False ):

              UI.show_leafs_only.render()

              UI.show_leafs_only.change(
                inputs = [ UI.project_name, UI.show_leafs_only ],
                outputs = UI.sample_tree,
                fn = lambda *args: gr.update( choices = get_samples(*args) ),
              )

                # UI.branch_sample_count.render()
                # UI.leaf_sample_count.render()

                # # Recount on sample_tree change
                # UI.sample_tree.change(
                #   inputs = UI.project_name,
                #   outputs = [ UI.branch_sample_count, UI.leaf_sample_count ],
                #   fn = lambda project_name: [
                #     len(get_samples(project_name, leafs_only)) for leafs_only in [ False, True ]
                #   ]
                # )
            
          UI.picked_sample.render()

          UI.sample_tree.change(
            inputs = [ UI.project_name, UI.sample_tree ],
            outputs = UI.picked_sample,
            fn = refresh_siblings,
            api_name = 'get-siblings'        
          )

          preview_inputs = [
              UI.project_name, UI.picked_sample, UI.cut_audio_specs, UI.preview_just_the_last_n_sec,
              UI.upsample_rendering, UI.combine_upsampling_levels, UI.invert_center_channel
          ]

          get_preview_args = lambda force_reload: dict(
            inputs = [
              *preview_inputs, gr.State(force_reload)
            ],
            outputs = [
              UI.sample_box, UI.current_chunks, #UI.generated_audio,
              UI.total_audio_length, UI.upsampled_lengths,
              UI.go_to_children_button, UI.go_to_parent_button,
              UI.picked_sample_updated
            ],
            fn = get_sample,
          )

          default_preview_args = get_preview_args(False)

          # Virtual input & handler to create an API method for get_sample_filename
          gr.Textbox(visible=False).change(
            inputs = preview_inputs,
            outputs = gr.Textbox(visible=False),
            fn = get_sample_filename,
            api_name = 'get-sample-filename'
          )

          UI.picked_sample.change(
            **default_preview_args,
            api_name = 'get-sample',
            _js =
            # Set the search string to ?[sample_id] for easier navigation
            '''
              async ( ...args ) => {

                try {

                  let sample_id = args[1]

                  sample_id && window.history.pushState( {}, '', `?${args[1]}` )

                  // Gray out the wavesurfer
                  Ji.grayOutWavesurfer()

                  // Now we'll try to reload the audio from cache. To do that, we'll find the first cached blob (Ji.blobCache) whose key starts with the sample_id either followed by space or end of string.
                  // (Although different version of the same sample might have been cached, the first one will be the one that was added last, so it's the most recent one)
                  let cached_blob = Ji.blobCache.find( ({ key }) => key.match( new RegExp(`^${sample_id}( |$)`) ) )
                  if ( cached_blob ) {
                    console.log( 'Found cached blob', cached_blob )
                    let { key, blob } = cached_blob
                    wavesurfer.loadBlob( blob )
                    Ji.lastLoadedBlobKey = key
                    Ji.preloadedAudio = true
                    // Gray out slightly less
                    Ji.grayOutWavesurfer( true, 0.75 )
                  }

                } catch (e) {
                  console.error(e)
                } finally {

                  return args

                }

              }
            '''
          )
           
          # When the picked sample is updated, update all the others too (UI.sibling_chunks) by calling get_sample for each sibling
          UI.picked_sample_updated.render().change(
            inputs = [ *preview_inputs ],
            outputs = UI.sibling_chunks,
            fn = get_sibling_samples,
            api_name = 'get-sibling-samples',
          )

          UI.current_chunks.render()
          UI.sibling_chunks.render()

          UI.upsampled_lengths.render().change(
            inputs = UI.upsampled_lengths,
            outputs = None,
            fn = None,
            # Split by comma and turn into floats and add wavesurfer markers for each (first clear all the markers)
            _js = 'comma_separated => Ji.addUpsamplingMarkers( comma_separated.split(",").map( parseFloat ) )'
          )

          with UI.sample_box.render():

            with UI.upsampling_accordion.render():

              with gr.Row():

                with gr.Column():

                  UI.upsampling_level.render().change(
                    **default_preview_args,
                  )

                  # Only show the upsampling elements if there are upsampled versions of the picked sample
                  def show_or_hide_upsampling_elements(project_name, sample_id, upsampling_running):

                    levels = get_levels(get_zs(project_name, sample_id))
                    # print(f'Levels: {levels}')

                    available_level_names = UI.UPSAMPLING_LEVEL_NAMES[:len(levels)]
                    print(f'Available level names: {available_level_names}')

                    return {
                      # UI.upsampling_accordion: gr.update(
                      #   visible = len(levels) > 1 or upsampling_running,
                      # ),
                      # (removing the accordion for now)
                      UI.upsampling_status: gr.update(
                        visible = upsampling_running,
                      ),
                      UI.upsampling_level: gr.update(
                        choices = available_level_names,
                        # Choose the highest available level
                        value = available_level_names[-1],
                      )
                    }
                  
                  show_or_hide_upsampling_elements_args = dict(
                    inputs = [ UI.project_name, UI.picked_sample, UI.upsampling_running ],
                    outputs = [ UI.upsampling_status, UI.upsampling_level ],
                    fn = show_or_hide_upsampling_elements,
                  )

                  UI.picked_sample.change( **show_or_hide_upsampling_elements_args )
                  UI.upsampling_running.change( **show_or_hide_upsampling_elements_args )

                with gr.Column() as upsampling_manipulation_column:

                  # # Show the column only if an upsampled sample is selected and hide the compose row respectively (we can only compose with the original sample)
                  # UI.upsampling_level.change(
                  #   inputs = [ UI.upsampling_level, UI.upsampling_running ],
                  #   outputs = [ upsampling_manipulation_column, UI.compose_row ],
                  #   fn = lambda upsampling_level, upsampling_running: [
                  #     gr.update( visible = upsampling_level != 'Raw' ),
                  #     gr.update( visible = upsampling_level == 'Raw' and not upsampling_running ),
                  #   ]
                  # )

                  with gr.Row():

                    UI.upsample_rendering.render().change(
                      **default_preview_args,
                    )

                    UI.combine_upsampling_levels.render().change(
                      **default_preview_args,
                    )

                    UI.invert_center_channel.render().change(
                      **default_preview_args,
                    )

              # Show the continue upsampling markdown only if the current level's length in tokens is less than the total audio length
              # Also update the upsampling button to say "Continue upsampling" instead of "Upsample"
              def show_or_hide_continue_upsampling(project_name, sample_id, total_audio_length, upsampling_running):

                if not upsampling_running:
                  zs = get_zs(project_name, sample_id)
                  levels = get_levels(zs)
                  # print(f'Levels: {levels}, z: {zs}')
                  # We'll show if there's no level 0 in levels or if the length of level 0 (in seconds) is less than the length of level 2 (in seconds)
                  must_show = 0 not in levels or tokens_to_seconds(len(zs[0]), 0) < tokens_to_seconds(len(zs[2]), 2)
                  # print(f'Must show: {must_show}')
                  
                else:
                  must_show = True

                return gr.update( visible = must_show )
              
              UI.picked_sample.change(
                inputs = [ UI.project_name, UI.picked_sample, UI.total_audio_length, UI.upsampling_running ],
                outputs = UI.continue_upsampling_button,
                fn = show_or_hide_continue_upsampling,
              )

              upsample_button_click_args = dict(
                inputs = UI.upsampling_running,
                outputs = [ UI.upsampling_running, UI.upsampling_triggered_by_button ],
                fn = lambda was_running: 
                # If was running (i.e. we're stopping), kill the runtime (after a warning) and show an alert saying to restart the runtime in Colab
                  [
                    print('Killing runtime...'),
                    subprocess.run(['kill', '-9', str(os.getpid())]),
                  ] if was_running else {
                    UI.upsampling_running: 1,
                    UI.upsampling_triggered_by_button: True,
                  },
                _js = """
                  // Confirm before starting/stopping the upsample process
                  running => {
                    confirmText = 
                      running ?
                        'Are you sure you want to stop the upsample process? ⚠️ THIS WILL KILL THE RUNTIME AND YOU WILL HAVE TO RESTART IT IN COLAB ⚠️ (But your current upsampling progress will be saved)' :
                        'Are you sure you want to start the upsample process? THIS WILL TAKE HOURS, AND YOU WON’T BE ABLE TO CONTINUE COMPOSING!'
                    if ( !confirm(confirmText) ) {
                      throw new Error(`${running ? 'Stopping' : 'Starting'} upsample process canceled by user`)
                    } else {
                      // If running, show a message saying to restart the runtime in Colab
                      if ( running ) {
                        alert('Upsample process stopped. Please re-run the cell in Colab to restart the UI')
                      }
                      return [ running ]
                    }
                  }
                """
              )

              UI.continue_upsampling_button.render().click( **upsample_button_click_args )

              UI.upsampling_audio_refresher.render()

              def reset_audio_refresher():
                Upsampling.should_refresh_audio = False

              [ 
                UI.upsampling_audio_refresher.change( **action ) for action in [ 
                  default_preview_args, 
                  show_or_hide_upsampling_elements_args,
                  dict(
                    inputs = None,
                    outputs = None,
                    # Reset Upsampling.should_refresh_audio to False
                    fn = reset_audio_refresher
                  )
                ] 
              ]

            UI.upsampling_status.render()
            
            # Refresh button
            internal_refresh_button = gr.Button('🔃', elem_id = 'internal-refresh-button', visible=False)
            
            internal_refresh_button.click(
              **get_preview_args(force_reload = True),
            )

            internal_refresh_button.click(
              **show_or_hide_upsampling_elements_args,
            )
                
            for element in [ 
              UI.audio_waveform,
              UI.audio_timeline
            ]:
              element.render()

            # Play/pause button, js-based
            gr.HTML("""
              <!-- Button to play/pause the audio -->
              <button class="gr-button gr-button-lg gr-button-secondary"
                onclick = "
                  wavesurfer.playPause()
                  this.innerText = wavesurfer.isPlaying() ? '⏸️' : '▶️'
                "
              >▶️</button>

              <!-- Textbox showing current time -->
              <input type="number" class="gr-box gr-input gr-text-input" id="audio-time" value="0">

              <!-- Download button -- it will be set to the right href later on -->
              <!--
              <a class="gr-button gr-button-lg gr-button-secondary" id="download-button">
                🔗
              </a>
              -->
              <!-- (Removed for now, as it only links to the first chunk, will fix later) -->

              <!-- Refresh button -- it virtually clicks the "internal-refresh-button" button (which is hidden) -->
              <button class="gr-button gr-button-lg gr-button-secondary" onclick="window.shadowRoot.getElementById('internal-refresh-button').click()" id="refresh-button">
                ↻
              </button>
            """)

            with UI.compose_row.render():

              gr.Button(
                value = 'Go on',
                variant = 'primary',
              ).click(
                inputs =  [ UI.project_name, UI.picked_sample, UI.show_leafs_only, *UI.generation_params ],
                outputs = [ UI.sample_tree, UI.generation_progress ],
                fn = generate,
              )

              gr.Button(
                value = 'More variations',          
              ).click(
                inputs = [ UI.project_name, UI.picked_sample, UI.show_leafs_only, *UI.generation_params ],
                outputs = [ UI.sample_tree, UI.generation_progress ],
                fn = lambda project_name, sample_id, *args: generate(project_name, get_parent(project_name, sample_id), *args),
              )

              UI.go_to_parent_button.render()
              UI.go_to_parent_button.click(
                inputs = [ UI.project_name, UI.picked_sample ],
                outputs = UI.sample_tree,
                fn = get_parent
              )

              UI.go_to_children_button.render()
              UI.go_to_children_button.click(
                inputs = [ UI.project_name, UI.picked_sample ], 
                outputs = UI.sample_tree,
                fn = lambda project_name, sample_id: get_children(project_name, sample_id)[0]
              )

              gr.Button('🗑️').click(
                inputs = [ UI.project_name, UI.picked_sample, gr.Checkbox(visible=False) ],
                outputs = [ UI.picked_sample, UI.sample_box ],
                fn = delete_sample,
                _js = """
                  ( project_name, child_sample_id ) => {
                    if ( confirm('Are you sure? There is no undo!') ) {
                      return [ project_name, child_sample_id, true ]
                    } else {
                      throw new Error('Cancelled; not deleting')
                    }
                  }
                """,
                api_name = 'delete-sample'
              )

            with gr.Accordion( 'Advanced', open = False ):

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

              with gr.Tab('Rename sample'):

                new_sample_id = gr.Textbox(
                  label = 'New sample id',
                  placeholder = 'Alphanumeric and dashes only'
                )

                gr.Button('Rename').click(
                  inputs = [ UI.project_name, UI.picked_sample, new_sample_id, UI.show_leafs_only ],
                  outputs = UI.sample_tree,
                  fn = rename_sample,
                  api_name = 'rename-sample'
                )
        
              with gr.Tab('Purge samples'):

                # For all samples whose parent sample's level 0/1 are the same as this one, purge those levels

                # We need a button to prepare the list of samples to purge, a multiline textbox to show the list, and a button to confirm the purge

                purge_list = gr.Textbox(
                  label = 'Purge list',
                  placeholder = 'Click the button below to prepare the list of samples to purge',
                  multiline = True,
                  disabled = True,
                )

                def prepare_purge_list(project_name):
                  # Get the list of samples to purge
                  # For each sample, get its parent sample
                  # If the parent sample's level 0/1 is the same as this one, add it to the list
                  samples = get_samples(project_name, False)
                  purge_list = []
                  for sample in samples:
                    try:
                      current_zs = t.load(get_zs_filename(project_name, sample))
                      parent = get_parent(project_name, sample)
                      if not parent:
                        print(f'No parent for {sample}, skipping')
                        continue
                      parent_zs = t.load(get_zs_filename(project_name, parent))
                      if ( 
                        t.equal( current_zs[0], parent_zs[0] ) and
                        t.equal( current_zs[1], parent_zs[1] ) and
                        ( current_zs[0].shape[1] + current_zs[1].shape[1] > 0 )
                      ):
                        purge_list.append(sample)
                        print(f'Adding {sample} to the purge list')
                    except Exception as e:
                      print(f'Error while processing {sample}: {e}, continuing')
                      continue
                  print(f'Purge list: {purge_list}')
                  return '\n'.join(purge_list)

                gr.Button('Prepare purge list').click(
                  inputs = [ UI.project_name ],
                  outputs = purge_list,
                  fn = prepare_purge_list,
                  api_name = 'prepare-purge-list'
                )

                def purge_samples(project_name, purge_list):
                  # Check the project size before purging, i.e. the joint size of all *.z files in the project directory
                  project_size_before = subprocess.check_output(['du','-sh', os.path.join( base_path, project_name )]).split()[0].decode('utf-8')
                  # For each sample in the purge list, delete it
                  for sample in purge_list.split('\n'):
                    zs = t.load(get_zs_filename(project_name, sample))
                    level2_shape0 = zs[2].shape[0]
                    purged_zs = [
                      t.empty([level2_shape0,0], device='cuda'),
                      t.empty([level2_shape0,0], device='cuda'),
                      zs[2]
                    ]
                    print(f'Purged {sample} from shapes {[x.shape for x in zs]} to {[x.shape for x in purged_zs]}')
                    t.save(purged_zs, get_zs_filename(project_name, sample))
                  # Check the project size after purging
                  project_size_after = subprocess.check_output(['du','-sh', os.path.join( base_path, project_name )]).split()[0].decode('utf-8')
                  return f'Project size before: {project_size_before}, after: {project_size_after}'
                
                gr.Button('Purge samples').click(
                  inputs = [ UI.project_name, purge_list ],
                  outputs = purge_list,
                  fn = purge_samples,
                  api_name = 'purge-samples'
                )

              with gr.Tab('Completify sample'):

                gr.Markdown('''
                  For space saving purposes, the app will sometime NOT include the entire information needed to render the sample into the sample file, taking the missing info (e.g. upsampled tokens) from its ancestors instead.
                            
                  If, for whatever reason, you want to have the entire information in the sample file, you can add it by clicking the button below.
                ''')

                def completify(project_name, sample_id):
                  zs = get_zs(project_name, sample_id, True)
                  backup_sample(project_name, sample_id)
                  save_zs(zs, project_name, sample_id)

                completify_button = gr.Button('Completify')
                completify_button.click(
                  completify,
                  [ UI.project_name, UI.picked_sample ],
                  gr.Button('Completify')
                )

        UI.generation_progress.render()

      with gr.Tab('Prime'):

        primed_audio_source = gr.Radio(
          label = 'Audio source',
          choices = [ 'microphone', 'upload' ],
          value = 'microphone'
        )

        UI.primed_audio.render()
        
        primed_audio_source.change(
          inputs = primed_audio_source,
          outputs = UI.primed_audio,
          fn = lambda source: gr.update( source = source ),
        )

        sec_to_trim_primed_audio = gr.Number(
          label = 'Trim starting audio to ... seconds from the beginning',
        )

        def trim_primed_audio(audio, sec):
          print(f'Trimming {audio} to {sec} seconds')
          # # Plot the audio to console for debugging
          # plt.plot(audio)
          # plt.show()              
          # Audio is of the form (sr, audio)
          trimmed_audio = audio[1][:int(sec * audio[0])]
          print(f'Trimmed audio shape is {trimmed_audio.shape}')
          return ( audio[0], trimmed_audio )

        sec_to_trim_primed_audio.submit(
          inputs = [ UI.primed_audio, sec_to_trim_primed_audio ],
          outputs = UI.primed_audio,
          fn = trim_primed_audio
        )

        prime_button = gr.Button(
          'Convert to sample',
          variant = 'primary'
        )
                
        prime_button.click(
          inputs = [ UI.project_name, UI.primed_audio, sec_to_trim_primed_audio, UI.show_leafs_only ],
          outputs = [ UI.sample_tree, prime_button, UI.prime_timestamp, UI.first_generation_row ], # UI.prime_timestamp is updated to the current time to force tab change
          fn = convert_audio_to_sample,
          api_name = 'convert-wav-to-sample'
        )

        UI.prime_timestamp.render().change(
          inputs = UI.prime_timestamp, outputs = None, fn = None,
          _js = 
            # Find a button inside a div inside another div with class 'tabs', the button having 'Workspace' as text, and click it -- all this in the shadow DOM.
            # Gosh, this is ugly.
            """
              timestamp => {
                console.log(`Timestamp changed to ${timestamp}; clicking the 'Workspace' tab`)
                Ji.clickTabWithText('Workspace')
                return timestamp
              }
            """
        )

      with gr.Tab('Upsample'):

        # Warning that this process is slow and can take up to 10 minutes for 1 second of audio
        with gr.Accordion('What is this?', open = False):

          gr.Markdown('''
            Upsampling is a process that creates higher-quality audio from your composition. It is done in two steps:

            - “Midsampling,” which considerably improves the quality of the audio, takes around 2 minutes per one second of audio.

            - “Upsampling,” which improves the quality some more, goes after midsampling and takes around 8 minutes per one second of audio.

            Thus, say, for a one-minute song, you will need to wait around 2 hours to have the midsampled version, and around 8 hours _more_ to have the upsampled version.

            You will be able to listen to the audio as it is being generated: Each “window” of upsampling takes ~6 minutes and will give you respectively ~2.7 and ~0.7 additional seconds of mid- or upsampled audio to listen to.

            ⚠️ WARNING: As the upsampling process uses a different model, which cannot be loaded together with the composition model due to memory constraints, **you will not be able to upsample and compose at the same time**. To go back to composing you will have to restart the Colab runtime or start a second Colab runtime and use them in parallel.
          ''')

        UI.sample_to_upsample.render()

        # Change the sample to upsample when a sample is picked
        UI.picked_sample.change(
          inputs = UI.picked_sample,
          outputs = UI.sample_to_upsample,
          fn = lambda x: x,
        )

        with gr.Accordion('Genres for upsampling (optional)', open = False):

          with gr.Accordion('What is this?', open = False):

            gr.Markdown('''
              The tool will generate three upsamplings of the selected sample, which will then be panned to the left, center, and right, respectively. Choosing different genres for each of the three upsamplings will result in a more diverse sound between them, thus enhancing the (pseudo-)stereo effect. 

              A good starting point is to have a genre that emphasizes vocals (e.g. `Pop`) for the center channel, and two similar but different genres for the left and right channels (e.g. `Rock` and `Metal`).

              If you don’t want to use this feature, simply select the same genre for all three upsamplings.
            ''')

          with gr.Row():

            for input in [ UI.genre_for_upsampling_left_channel, UI.genre_for_upsampling_center_channel, UI.genre_for_upsampling_right_channel ]:

              input.render()
          
          UI.kill_runtime_once_done.render()
        
        # If upsampling is running, enable the upsampling_refresher -- a "virtual" input that, when changed, will update the upsampling_status_markdown
        # It will do so after waiting for 10 seconds (using js). After finishing, it will update itself again, causing the process to repeat.
        UI.upsampling_refresher.render().change(
          inputs = [ UI.upsampling_refresher, UI.upsampling_audio_refresher ],
          outputs = [ UI.upsampling_refresher, UI.upsampling_status, UI.upsampling_audio_refresher ],
          fn = lambda refresher, audio_refresher: {
            UI.upsampling_status: Upsampling.status_markdown,
            UI.upsampling_refresher: refresher + 1,
            UI.upsampling_audio_refresher: audio_refresher + 1 if Upsampling.should_refresh_audio else audio_refresher
          },
          _js = """
            async ( ...args ) => {
              await new Promise( resolve => setTimeout( resolve, 10000 ) )
              console.log( 'Checking upsampling status...' )
              return args
            }
          """
        )

        UI.upsample_button.render().click( **upsample_button_click_args )

        # During app load, set upsampling_running and upsampling_stopping according to Upsampling.running
        app.load(
          inputs = None,
          outputs = UI.upsampling_running,
          fn = lambda: Upsampling.running,
        )
        
        UI.upsampling_triggered_by_button.render()

        # When upsampling_running changes via the button, run the upsampling process
        UI.upsampling_running.render().change(
          inputs = [
            UI.upsampling_triggered_by_button,
            UI.project_name, UI.sample_to_upsample, UI.artist, UI.lyrics,
            UI.genre_for_upsampling_left_channel, UI.genre_for_upsampling_center_channel, UI.genre_for_upsampling_right_channel,
            UI.kill_runtime_once_done            
          ],
          outputs = None,
          fn = lambda triggered_by_button, *args: start_upsampling( *args ) if triggered_by_button else None,
          api_name = 'toggle-upsampling',
          # Also go to the "Workspace" tab (because that's where we'll display the upsampling status) via the Ji.clickTabWithText helper method in js
          _js = """
            async ( ...args ) => {
              console.log( 'Upsampling toggled, args:', args )
              if ( args[0] ) {
                Ji.clickTabWithText( 'Workspace' )
                return args
              } else {
                throw new Error('Upsampling not triggered by button')
              }
            }
          """
        )

        # When it changes regardless of the session, e.g. also at page refresh, update the various relevant UI elements, start the refresher, etc.
        UI.upsampling_running.change(
          inputs = None,
          outputs = [ UI.upsampling_status, UI.upsample_button, UI.continue_upsampling_button, UI.upsampling_refresher ],
          fn = lambda: {
            UI.upsampling_status: 'Upsampling in progress...',
            UI.upsample_button: gr.update(
              value = 'Stop upsampling',
              variant = 'secondary',
            ),
            UI.continue_upsampling_button: gr.update(
              value = 'Stop upsampling',
            ),
            # Random refresher value (int) to trigger the refresher
            UI.upsampling_refresher: random.randint( 0, 1000000 ),
            # # Hide the compose row
            # UI.compose_row: HIDE,
          }
        )

      with gr.Tab('Panic'):

        with gr.Accordion('What is this?', open = False):

          gr.Markdown('''
            Sometimes the app will crash due to insufficient GPU memory. If this happens, you can try using the button below to empty the cache. Usually around 12 GB of GPU RAM is needed to safely run the app.

            If that doesn’t work, you’ll have to restart the runtime (`Runtime` > `Restart and run all` in Colab). That’ll take a couple of minutes, but the memory will be new as a daisy.
          ''')

        memory_usage = gr.Textbox(
          label = 'GPU memory usage',
          value = 'Click Refresh to update',
        )

        def get_gpu_memory_usage():
          return subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader'],
            encoding='utf-8'
          ).strip()

        with gr.Row():
        
          gr.Button('Refresh').click(
            inputs = None,
            outputs = memory_usage,
            fn = get_gpu_memory_usage,
            api_name = 'get-gpu-memory-usage',
          )

          gr.Button('Empty cache', variant='primary').click(
            inputs = None,
            outputs = memory_usage,
            fn = lambda: [
              empty_cache(),
              get_gpu_memory_usage(),
            ][-1],
            api_name = 'empty-cache',
          )

        with gr.Accordion('Dev', open = False, visible = DEV_MODE):

          # Button to kill current process
          gr.Button('Kill current process').click(
            inputs = None,
            outputs = None,
            fn = lambda: subprocess.run( ['kill', '-9', str(os.getpid())] ),
            api_name = 'kill-current-process',
          )

          gr.Markdown('''
            The following input box allows you to execute arbitrary Python code. ⚠️ DON’T USE THIS FEATURE IF YOU DON’T KNOW WHAT YOU’RE DOING! ⚠️
          ''')

          eval_server_code = gr.Textbox(
            label = 'Python code',
            placeholder = 'Shift+Enter for a new line, Enter to run',
            value = '',
            max_lines = 10,
          )

          eval_button = gr.Button('Execute')

          eval_output = gr.Textbox(
            label = 'Output',
            value = '',
            max_lines = 10,
          )

          eval_args = dict(
            inputs = eval_server_code,
            outputs = eval_output,
            fn = lambda code: {
              eval_output: eval( code )
            }
          )

          eval_button.click(**eval_args)
          eval_server_code.submit(
            **eval_args,
            api_name = 'eval-code',
          )

  # TODO: Don't forget to remove this line before publishing the app
  frontend_on_load_url = f'https://cdn.jsdelivr.net/gh/vzakharov/jukebox-webui@{GITHUB_SHA}/frontend-on-load.js'
  with urllib.request.urlopen(frontend_on_load_url) as response:
    frontend_on_load_js = response.read().decode('utf-8')

    try:
      old_frontend_on_load_md5 = frontend_on_load_md5
    except NameError:
      old_frontend_on_load_md5 = None

    frontend_on_load_md5 = hashlib.md5(frontend_on_load_js.encode('utf-8')).hexdigest()
    print(f'Loaded frontend-on-load.js from {response.geturl()}, md5: {frontend_on_load_md5}')

    if frontend_on_load_md5 != old_frontend_on_load_md5:
      print('(New version)')
    else:
      print('(Same version as during the previous run)')

    # print(frontend_on_load_js)

  app.load(
    on_load,
    inputs = [ gr.Textbox(visible=False), gr.Textbox(visible=False), gr.Textbox(visible=False) ],
    outputs = [ 
      UI.project_name, UI.routed_sample_id, UI.artist, UI.genre_dropdown, UI.getting_started_column, UI.separate_tab_warning, UI.separate_tab_link, UI.main_window,
      UI.genre_for_upsampling_left_channel, UI.genre_for_upsampling_center_channel, UI.genre_for_upsampling_right_channel
    ],
    api_name = 'initialize',
    _js = frontend_on_load_js,
    # _js = """
    # // (insert manually for debugging)
    # """,
  )

  # Also load browser's time zone offset on app load
  def set_browser_timezone(offset):
    global browser_timezone

    print('Browser time zone offset:', offset)
    browser_timezone = timezone(timedelta(minutes = -offset))
    print('Browser time zone:', browser_timezone)

  app.load(
    inputs = gr.Number( visible = False ),
    outputs = None,
    _js = '() => [ new Date().getTimezoneOffset() ]',
    fn = set_browser_timezone
  )

  app.launch( share = share_gradio, debug = debug_gradio )