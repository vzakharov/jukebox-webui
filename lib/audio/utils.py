def trim_primed_audio(audio, sec):
  print(f'Trimming {audio} to {sec} seconds')
  # # Plot the audio to console for debugging
  # plt.plot(audio)
  # plt.show()              
  # Audio is of the form (sr, audio)
  trimmed_audio = audio[1][:int(sec * audio[0])]
  print(f'Trimmed audio shape is {trimmed_audio.shape}')
  return ( audio[0], trimmed_audio )