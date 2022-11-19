import sys
sys.path.append("..")
import librosa
import audio as Audio
import torch

def preprocess_audio(audio_file, _stft):
  wav, sample_rate = librosa.load(audio_file, sr=22050)
  mel_spectrogram, _ = Audio.tools.get_mel_from_wav(wav, _stft)
  return torch.from_numpy(mel_spectrogram).to(device=device)