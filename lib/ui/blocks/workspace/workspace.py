

from UI.main import workspace_column
from .tabs.upsample.upsample import render_upsample_tab

from .tabs.main import render_main_workspace_tab
from .tabs.prime import render_prime_tab
from .tabs.panic.panic import render_panic_tab

def render_workspace():
  with workspace_column.render():

    render_main_workspace_tab()

    render_prime_tab()

    render_upsample_tab()

    render_panic_tab()

