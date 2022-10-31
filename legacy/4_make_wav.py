if step == 'MAKE WAV':
  try:
    del top_prior
    empty_cache()
  except:
    pass
  id = '' #@param{type:'string'}
  level = "0" #@param[0,1,2]
  level = int(level)
  zs_to_wavs(id, level=level)
else:
  print('skipping MAKE WAV')