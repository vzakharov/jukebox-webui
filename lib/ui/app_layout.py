from params import DEV_MODE, GITHUB_SHA


import gradio as gr


def app_layout():
    return gr.Blocks(
  css = """
    .gr-button {
      /* add margin to the button */
      margin: 5px 5px 5px 5px;
    }
    #getting-started-column {
      /* add a considerable margin to the left of the column */
      margin-left: 20px;
    }
    #generation-progress {
      /* gray, smaller font */
      color: #777;
      font-size: 0.8rem;
    }
    #audio-timeline {
      /* hide for now */
      display: none;
    }
  """,
  title = f'Jukebox Web UI { GITHUB_SHA }{ " (dev mode)" if DEV_MODE else "" }',
)