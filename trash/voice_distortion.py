from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift, TanhDistortion
import numpy as np
import librosa
from scipy.io.wavfile import write
# from audiomentations import TanhDistortion


# augment = Compose([
#     AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
#     TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
#     PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
#     Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
# ])

transform = TanhDistortion(
    min_distortion=0.01,
    max_distortion=0.7,
    p=1.0
)

samples, _ = librosa.load("./audio_ref/chung_jaist.wav", sr=22050)

# Augment/transform/perturb the audio data
augmented_samples = transform(samples=samples, sample_rate=22050)
write("./wav_output/distortion.wav", 22050, augmented_samples)