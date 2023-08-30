

from lib.ui.UI import UI
from lib.ui.components.workspace.tabs.upsample.upsample import render_upsample_tab

from .tabs.main import render_main_workspace_tab
from .tabs.prime import render_prime_tab
from .tabs.panic.panic import render_panic_tab

def render_workspace(app):
  with UI.workspace_column.render():

    render_main_workspace_tab(app)

    render_prime_tab()

    render_upsample_tab(app)

    render_panic_tab()

