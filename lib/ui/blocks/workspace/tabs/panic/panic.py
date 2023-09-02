from .dev import render_dev_accordion

import gradio as gr
from jukebox.utils.torch_utils import empty_cache

from lib.utils import get_gpu_memory_usage

def render_panic_tab():
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

    render_dev_accordion()