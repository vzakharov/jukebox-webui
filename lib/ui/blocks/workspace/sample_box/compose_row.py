from lib.model.generate import generate
from lib.navigation.delete_sample import delete_sample
from lib.navigation.get_children import get_children
from lib.navigation.get_parent import get_parent

import gradio as gr

from UI.general import project_name, project_name, project_name, project_name, project_name
from UI.misc import generation_progress, generation_progress
from UI.navigation import picked_sample, show_leafs_only, sample_tree, picked_sample, show_leafs_only, sample_tree, picked_sample, sample_tree, picked_sample, sample_tree, picked_sample, picked_sample, sample_box
from UI.project import generation_params, generation_params
from UI.sample import compose_row, go_to_parent, go_to_parent, go_to_children, go_to_children

def render_compose_row():
  with compose_row.render():
    gr.Button(
      value = 'Go on',
      variant = 'primary',
    ).click(
      inputs =  [ project_name, picked_sample, show_leafs_only, *generation_params ],
      outputs = [ sample_tree, generation_progress ],
      fn = generate,
    )

    gr.Button(
      value = 'More variations',
    ).click(
      inputs = [ project_name, picked_sample, show_leafs_only, *generation_params ],
      outputs = [ sample_tree, generation_progress ],
      fn = lambda project_name, sample_id, *args: generate(project_name, get_parent(project_name, sample_id), *args),
    )

    go_to_parent.render()
    go_to_parent.click(
      inputs = [ project_name, picked_sample ],
      outputs = sample_tree,
      fn = get_parent
    )

    go_to_children.render()
    go_to_children.click(
      inputs = [ project_name, picked_sample ],
      outputs = sample_tree,
      fn = lambda project_name, sample_id: get_children(project_name, sample_id)[0]
    )

    gr.Button('🗑️').click(
      inputs = [ project_name, picked_sample, gr.Checkbox(visible=False) ],
      outputs = [ picked_sample, sample_box ],
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