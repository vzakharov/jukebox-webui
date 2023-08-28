from lib.upsampling.Upsampling import Upsampling

def request_to_stop_upsampling():
  if Upsampling.running:
    print('Stopping upsampling...')
    Upsampling.stop = True
  else:
    print('No upsampling running')