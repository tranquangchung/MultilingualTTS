import auditok
import pdb
import wave
import librosa
import numpy as np
from scipy.io.wavfile import write

wav, sr = librosa.load("SSB18370408.wav")
max_wav_value=32768
wav = wav / max(abs(wav)) * max_wav_value
write(f"SSB18370408_test.wav", sr, wav.astype(np.int16))

# split returns a generator of AudioRegion objects
audio_regions = auditok.split(
    "SSB18370408_test.wav",
    min_dur=2,     # minimum duration of a valid audio event in seconds
    max_dur=3,       # maximum duration of an event
    max_silence=0.3, # maximum duration of tolerated continuous silence within an event
    energy_threshold=55 # threshold of detection
)
audio_file = [audio for audio in audio_regions]
print(len(audio_file))
# for i, r in enumerate(audio_regions):
#     # Regions returned by `split` have 'start' and 'end' metadata fields
#     print("Region {i}: {r.meta.start:.3f}s -- {r.meta.end:.3f}s".format(i=i, r=r))
#     filename = r.save("region_{meta.start:.3f}-{meta.end:.3f}.wav")
#     print("region saved as: {}".format(filename))