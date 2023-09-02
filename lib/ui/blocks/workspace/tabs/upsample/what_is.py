what_is_upsampling_markdown = '''
  Upsampling is a process that creates higher-quality audio from your composition. It is done in two steps:
  - “Midsampling,” which considerably improves the quality of the audio, takes around 2 minutes per one second of audio.
  - “Upsampling,” which improves the quality some more, goes after midsampling and takes around 8 minutes per one second of audio.
  Thus, say, for a one-minute song, you will need to wait around 2 hours to have the midsampled version, and around 8 hours _more_ to have the upsampled version.
  You will be able to listen to the audio as it is being generated: Each “window” of upsampling takes ~6 minutes and will give you respectively ~2.7 and ~0.7 additional seconds of mid- or upsampled audio to listen to.
  ⚠️ WARNING: As the upsampling process uses a different model, which cannot be loaded together with the composition model due to memory constraints, **you will not be able to upsample and compose at the same time**. To go back to composing you will have to restart the Colab runtime or start a second Colab runtime and use them in parallel.
'''