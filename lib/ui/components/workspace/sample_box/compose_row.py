from lib.generate import generate
from lib.navigation.delete_sample import delete_sample
from lib.navigation.get_children import get_children
from lib.navigation.get_parent import get_parent
from lib.ui.UI import UI

import gradio as gr

def render_compose_row():
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