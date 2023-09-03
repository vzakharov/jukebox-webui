from params import DEV_MODE

import gradio as gr

import os
import subprocess

def render_dev_accordion():
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