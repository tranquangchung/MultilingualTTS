import pyworld as pw
import librosa
import numpy as np
import pdb

wav, fs = librosa.load("/home/ldap-users/s2220411/Code/FastSpeech2_multilingual/Recognition/wav1.wav")
_f0, t = pw.dio(wav.astype(np.float64), fs)    # raw pitch extractor
f0 = pw.stonemask(wav.astype(np.float64), _f0, t, fs)  # pitch refinement
pdb.set_trace()
print(_f0)
