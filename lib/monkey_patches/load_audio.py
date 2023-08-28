# Monkey patch load_audio, allowing for duration = None
import librosa


def monkey_patched_load_audio(file, sr, offset, duration, mono=False):
  # Librosa loads more filetypes than soundfile
  x, _ = librosa.load(file, sr=sr, mono=mono, offset=offset/sr, duration=None if duration is None else duration/sr)
  if len(x.shape) == 1:
      x = x.reshape((1, -1))
  return x