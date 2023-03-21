import glob
import librosa
import pdb
import shutil

index = 0
path_duration = "/home/ldap-users/s2220411/Code/FastSpeech2_multilingual/visualization/duration_toolong"
fileaudio = glob.glob("/home/ldap-users/s2220411/Code/FastSpeech2_multilingual/visualization/synthesis_audio_interspeech2023_final1/*.wav")
for filename in fileaudio:
  y, sr = librosa.load(filename)
  name = filename.split("/")[-1]
  duration = y.shape[0] / sr
  if duration > 12:
    index += 1
    shutil.copy2(filename, path_duration)
    print(name, "duration: ", duration)
print(index)
