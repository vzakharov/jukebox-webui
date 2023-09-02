from datetime import timedelta

class Upsampling:

  project = None
  sample_id = None

  running = False
  zs = None
  level = None
  metas = None
  labels = None
  priors = None
  params = None

  windows = []
  window_index = 0
  window_start_time = None
  # Set time per window by default to 6 minutes (will be updated later) in timedelta format
  time_per_window = timedelta(minutes=6)
  windows_remaining = None
  time_remaining = None
  eta = None

  status_markdown = None
  should_refresh_audio = False

  stop = False
  kill_runtime_once_done = False

  upsamplers = None