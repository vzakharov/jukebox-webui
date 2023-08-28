import re
from main import browser_timezone
from lib.model.params import chunk_size, hps, raw_to_tokens


def as_local_hh_mm(dt, include_seconds = False):
  return dt.astimezone(browser_timezone).strftime('%H:%M:%S' if include_seconds else '%H:%M')


def convert_name(name):
  return re.sub(r'[^a-z0-9]+', '-', name.lower())


# else:
#   print('Settings are the same as loaded settings, not saving.')

def seconds_to_tokens(sec, level = 2):

  global hps, raw_to_tokens, chunk_size

  tokens = sec * hps.sr // raw_to_tokens
  tokens = ( (tokens // chunk_size) + 1 ) * chunk_size

  # For levels 1 and 0, multiply by 4 and 16 respectively
  tokens *= 4 ** (2 - level)

  return int(tokens)


def tokens_to_seconds(tokens, level = 2):

  global hps, raw_to_tokens

  return tokens * raw_to_tokens / hps.sr / 4 ** (2 - level)

