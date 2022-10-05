if step == 'UPSAMPLE':

  filename = '' #@param{type:'string'}
  choice = 1 #@param{type:'number'}
  choice -= 1
  cut_from_seconds = 0 #@param{type:'number'}

  if filename:
    zs = t.load(f'{hps.name}/{filename}.zs')
  else:
    assert zs is not None, 'No filename given and no zs loaded'
    
  print(f'Loaded zs of shape {zs[2].shape} from {filename}.zs')

  if cut_from_seconds:
    cut_from_tokens = seconds_to_tokens(cut_from_seconds, hps.sr, top_prior, chunk_size)
    print(f'Cutting from {cut_from_seconds} seconds ({cut_from_tokens} tokens)')
    zs[2] = zs[2][:, cut_from_tokens:]
    print(f'New zs shape: {zs[2].shape}')
    metas[0]['offset'] = cut_from_tokens

  # We always upsample 3 samples

  zs[2] = zs[2][choice].repeat(3, 1)
  zs[0] = t.empty((3,0), dtype=t.int64).cuda()
  zs[1] = t.empty((3,0), dtype=t.int64).cuda()

  # Make metas[2] and metas[1] same as metas[0] (shallow copy)
  metas = [metas[0], dict(metas[0]), dict(metas[0])]

  # Set genre to Metalcore for metas[0], Rock for metas[1] and Metal for metas[2]
  metas[0]['genre'] = 'Metalcore' #@param {type:'string'}
  metas[1]['genre'] = 'Rock' #@param {type:'string'}
  metas[2]['genre'] = 'Metal' #@param {type:'string'}

  labels = top_prior.labeller.get_batch_labels(metas, 'cuda')

  assert zs[2].shape[1]>=2048, f'Please first generate at least 2048 tokens at the top level, currently you have {zs[2].shape[1]}'
  hps.sample_length = zs[2].shape[1]*raw_to_tokens

  del top_prior
  empty_cache()
  top_prior=None

  upsamplers = [ make_prior(setup_hparams(prior, dict()), vqvae, 'cpu') for prior in priors[:-1] ]

  sampling_kwargs = [dict(temp=.99, fp16=True, max_batch_size=16, chunk_size=32),
                      dict(temp=0.99, fp16=True, max_batch_size=16, chunk_size=32),
                      None]

  if type(labels)==dict:
    labels = [prior.labeller.get_batch_labels(metas, 'cuda') for prior in upsamplers] + [labels]

  """This next step upsamples 2 levels.  The level_1 samples will be available after around one hour (depending on the length of your sample) and are saved under {hps.name}/level_0/item_0.wav, while the fully upsampled level_0 will likely take 4-12 hours. You can access the wav files down below, or using the "Files" panel at the left of this CoLab.

  (Please note, if you are using this CoLab on Google's free tier, you may want to download intermediate steps as the connection will last for a maximum 12 hours.)
  """

  # print(hps)

  print(sampling_kwargs)
  zs = upsample(zs, labels, sampling_kwargs, [*upsamplers, top_prior], hps)
else:
  print('skipping UPSAMPLE')