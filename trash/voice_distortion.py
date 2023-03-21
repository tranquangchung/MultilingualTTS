from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
from audiomentations import BandPassFilter, BandStopFilter, ClippingDistortion, RoomSimulator, TanhDistortion
import numpy as np
import librosa
from scipy.io.wavfile import write
from audiomentations import TanhDistortion
import numpy as np
import pdb
import time


augment = Compose([
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
    ClippingDistortion(),
    TanhDistortion()
])

#
samples, _ = librosa.load("./audio_ref/chung_ipad.wav", sr=22050)
#
# # Augment/transform/perturb the audio data
augmented_samples = augment(samples=samples, sample_rate=22050)
augmented_samples = augmented_samples / max(abs(augmented_samples))
write("./wav_output/distortion.wav", 22050, augmented_samples)