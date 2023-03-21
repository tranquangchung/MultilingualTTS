import librosa
import glob
import os
from scipy.io.wavfile import write
import numpy as np
import tqdm
import pdb

path = "/home/ldap-users/Share/Corpora/Speech/multi/VTCK/wav48_silence_trimmed"
path2save = "/home/ldap-users/Share/Corpora/Speech/multi/VTCK/downsampled_wavs"
speakers = os.listdir(path)
max_value = 32767
for speaker in tqdm.tqdm(speakers):
  if speaker == "log.txt": continue
  os.makedirs(os.path.join(path2save, speaker), exist_ok=True)
  wav_paths = glob.glob(os.path.join(path, speaker, "*"))
  for wav_path in wav_paths:
    wav_audio, sr = librosa.load(wav_path)
    wav_audio = wav_audio / max(abs(wav_audio)) * max_value
    filename = wav_path.split("/")[-1].split(".")[0].replace("_mic1", "").replace("_mic2", "")
    filename_path = os.path.join(path2save, speaker, filename+".wav")
    write(filename_path, sr, wav_audio.astype(np.int16))