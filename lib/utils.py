import re
from main import browser_timezone


def as_local_hh_mm(dt, include_seconds = False):
  return dt.astimezone(browser_timezone).strftime('%H:%M:%S' if include_seconds else '%H:%M')


def convert_name(name):
  return re.sub(r'[^a-z0-9]+', '-', name.lower())


