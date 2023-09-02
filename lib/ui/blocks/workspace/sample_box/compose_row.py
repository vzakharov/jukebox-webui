from lib.model.generate import generate
from lib.navigation.delete_sample import delete_sample
from lib.navigation.get_children import get_children
from lib.navigation.get_parent import get_parent

import gradio as gr

import lib.ui.components.general
import lib.ui.components.misc
import lib.ui.components.navigation
import lib.ui.components.project
import lib.ui.components.sample

def render_compose_row():
  with lib.ui.components.sample.compose_row.render():
    gr.Button(
      value = 'Go on',
      variant = 'primary',
    ).click(
      inputs =  [ lib.ui.components.general.project_name, lib.ui.components.navigation.picked_sample, lib.ui.components.navigation.show_leafs_only, *lib.ui.components.project.generation_params ],
      outputs = [ lib.ui.components.navigation.sample_tree, lib.ui.components.misc.generation_progress ],
      fn = generate,
    )

    gr.Button(
      value = 'More variations',
    ).click(
      inputs = [ lib.ui.components.general.project_name, lib.ui.components.navigation.picked_sample, lib.ui.components.navigation.show_leafs_only, *lib.ui.components.project.generation_params ],
      outputs = [ lib.ui.components.navigation.sample_tree, lib.ui.components.misc.generation_progress ],
      fn = lambda project_name, sample_id, *args: generate(project_name, get_parent(project_name, sample_id), *args),
    )

    lib.ui.components.sample.go_to_parent_button.render()
    lib.ui.components.sample.go_to_parent_button.click(
      inputs = [ lib.ui.components.general.project_name, lib.ui.components.navigation.picked_sample ],
      outputs = lib.ui.components.navigation.sample_tree,
      fn = get_parent
    )

    lib.ui.components.sample.go_to_children_button.render()
    lib.ui.components.sample.go_to_children_button.click(
      inputs = [ lib.ui.components.general.project_name, lib.ui.components.navigation.picked_sample ],
      outputs = lib.ui.components.navigation.sample_tree,
      fn = lambda project_name, sample_id: get_children(project_name, sample_id)[0]
    )

    gr.Button('🗑️').click(
      inputs = [ lib.ui.components.general.project_name, lib.ui.components.navigation.picked_sample, gr.Checkbox(visible=False) ],
      outputs = [ lib.ui.components.navigation.picked_sample, lib.ui.components.navigation.sample_box ],
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