import urllib.request

lists = {}

def get_list(what):
  items = []
  # print(f'Getting {what} list...')
  # If list already exists, return it
  if what in lists:
    # print(f'{what} list already exists.')
    return lists[what]
  else:
    with urllib.request.urlopen(f'https://raw.githubusercontent.com/openai/jukebox/master/jukebox/data/ids/v2_{what}_ids.txt') as f:
      for line in f:
        item = line.decode('utf-8').split(';')[0]
        item = item.replace('_', ' ').title()
        items.append(item)
    items.sort()
    print(f'Loaded {len(items)} {what}s.')
    lists[what] = items
  return items