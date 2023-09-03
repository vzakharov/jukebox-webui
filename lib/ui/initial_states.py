import gradio as gr

from lib.lists import get_list

from .elements.main import main_window
from .elements.general import project_name as project_name_element
from .elements.metas import artist, genre_dropdown
from .elements.misc import getting_started_column, separate_tab_warning
from .elements.navigation import routed_sample_id
from .elements.upsampling import (genre_center_channel, genre_left_channel,
                                  genre_right_channel)


def initial_states(projects, project_name, sample_id):
  return {
    project_name_element: gr.update(
      choices = projects,
      value = project_name,
    ),
    routed_sample_id: sample_id,
    artist: gr.update(
      choices = get_list('artist'),
    ),
    genre_dropdown: gr.update(
      choices = get_list('genre'),
    ),
    getting_started_column: gr.update(
      visible = len(projects) == 1
    ),
    separate_tab_warning: gr.update(
      visible = False
    ),
    main_window: gr.update(
      visible = True
    ),
    genre_left_channel: gr.update(
      choices = get_list('genre')
    ),
    genre_center_channel: gr.update(
      choices = get_list('genre'),
    ),
    genre_right_channel: gr.update(
      choices = get_list('genre'),
    ),
  }