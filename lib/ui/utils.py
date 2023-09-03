import gradio as gr

SHOW = gr.update( visible = True )
HIDE = gr.update( visible = False )
SHOW_OR_HIDE = lambda x: gr.update( visible = x )

browser_timezone = None