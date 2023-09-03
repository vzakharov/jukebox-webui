import os

def split_into_chunks(filename):
  chunk_filenames = []

  # If the mp3 size is > certain size, we'll need to send it back in chunks, so we divide the mp3 into as many chunks as needed
  file_size = os.path.getsize(f'{filename}.mp3')
  file_limit = 300000

  if file_size > file_limit:
    print(f'MP3 file size is {file_size} bytes, splitting into chunks...')
    file_content = open(f'{filename}.mp3', 'rb').read()
    for i in range(0, file_size, file_limit):
    # Place the chunk file in tmp/[filename without path] [range].mp3_chunk
    # Create the tmp folder if it doesn't exist
      if not os.path.exists(f'tmp'):
        os.makedirs(f'tmp')
      chunk_filename = f'tmp/{os.path.basename(filename)}{i}-{i+file_limit} .mp3_chunk'
      with open(chunk_filename, 'wb') as f:
        f.write(file_content[i:i+file_limit])
        print(f'Wrote bytes {i}-{i+file_limit} to {chunk_filename}')
      chunk_filenames.append(chunk_filename)
  else:
    chunk_filenames = [f'{filename}.mp3']

  print(f'Files to send: {chunk_filenames}')
  return chunk_filenames