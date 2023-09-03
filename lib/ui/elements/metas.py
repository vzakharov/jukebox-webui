### Project-specific

## Metas (artist, genre, lyrics)
import gradio as gr

artist = gr.Dropdown(
  label = 'Artist'
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
genre_dropdown = gr.Dropdown(
  label = 'Available genres'
)