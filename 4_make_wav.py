if step == 'MAKE WAV':
  id = 'nihil-0-0' #@param{type:'string'}
  filename = f'{hps.name}/{project_name}/{id}.zs'
  zs_to_wavs(filename)
else:
  print('skipping MAKE WAV')