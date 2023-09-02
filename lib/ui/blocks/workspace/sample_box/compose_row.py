from lib.model.generate import generate
from lib.navigation.delete_sample import delete_sample
from lib.navigation.get_children import get_children
from lib.navigation.get_parent import get_parent

import gradio as gr

import UI.general
import UI.misc
import UI.navigation
import UI.project
import UI.sample

def render_compose_row():
  with UI.sample.compose_row.render():
    gr.Button(
      value = 'Go on',
      variant = 'primary',
    ).click(
      inputs =  [ UI.general.project_name, UI.navigation.picked_sample, UI.navigation.show_leafs_only, *UI.project.generation_params ],
      outputs = [ UI.navigation.sample_tree, UI.misc.generation_progress ],
      fn = generate,
    )

    gr.Button(
      value = 'More variations',
    ).click(
      inputs = [ UI.general.project_name, UI.navigation.picked_sample, UI.navigation.show_leafs_only, *UI.project.generation_params ],
      outputs = [ UI.navigation.sample_tree, UI.misc.generation_progress ],
      fn = lambda project_name, sample_id, *args: generate(project_name, get_parent(project_name, sample_id), *args),
    )

    UI.sample.go_to_parent.render()
    UI.sample.go_to_parent.click(
      inputs = [ UI.general.project_name, UI.navigation.picked_sample ],
      outputs = UI.navigation.sample_tree,
      fn = get_parent
    )

    UI.sample.go_to_children.render()
    UI.sample.go_to_children.click(
      inputs = [ UI.general.project_name, UI.navigation.picked_sample ],
      outputs = UI.navigation.sample_tree,
      fn = lambda project_name, sample_id: get_children(project_name, sample_id)[0]
    )

    gr.Button('🗑️').click(
      inputs = [ UI.general.project_name, UI.navigation.picked_sample, gr.Checkbox(visible=False) ],
      outputs = [ UI.navigation.picked_sample, UI.navigation.sample_box ],
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