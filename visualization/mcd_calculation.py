# import some stuff
import os
import math
import glob
import librosa
import pyworld
import pysptk
import numpy as np
import matplotlib.pyplot as plot
import pdb

def load_wav(wav_file, sr):
  """
  Load a wav file with librosa.
  :param wav_file: path to wav file
  :param sr: sampling rate
  :return: audio time series numpy array
  """
  wav, _ = librosa.load(wav_file, sr=sr, mono=True)
  return wav

def log_spec_dB_dist(x, y):
  log_spec_dB_const = 10.0 / math.log(10.0) * math.sqrt(2.0)
  diff = x - y
  return log_spec_dB_const * math.sqrt(np.inner(diff, diff))

SAMPLING_RATE = 22050
FRAME_PERIOD = 5.0

alpha = 0.65  # commonly used at 22050 Hz
fft_size = 512
mcep_size = 34

def wav2mcep_numpy(wavfile, target_directory, alpha=0.65, fft_size=512, mcep_size=34):
  # make relevant directories
  if not os.path.exists(target_directory):
    os.makedirs(target_directory, exist_ok=True)
  loaded_wav = load_wav(wavfile, sr=SAMPLING_RATE)
  # Use WORLD vocoder to spectral envelope
  _, sp, _ = pyworld.wav2world(loaded_wav.astype(np.double), fs=SAMPLING_RATE,
                               frame_period=FRAME_PERIOD, fft_size=fft_size)
  # Extract MCEP features
  mgc = pysptk.sptk.mcep(sp, order=mcep_size, alpha=alpha, maxiter=0,
                         etype=1, eps=1.0E-8, min_det=0.0, itype=3)
  fname = os.path.basename(wavfile).split('.')[0]
  np.save(os.path.join(target_directory, fname + '.npy'), mgc, allow_pickle=False)

def average_mcd(ref_mcep_files, synth_mcep_files, cost_function):
  """
  Calculate the average MCD.
  :param ref_mcep_files: list of strings, paths to MCEP target reference files
  :param synth_mcep_files: list of strings, paths to MCEP converted synthesised files
  :param cost_function: distance metric used
  :returns: average MCD, total frames processed
  """
  min_cost_tot = 0.0
  frames_tot = 0
  filenames = glob.glob(os.path.join(ref_mcep_files, "*"))
  for filename in filenames:
    name = filename.split("/")[-1]
    ref_vec = np.load(filename)
    synthesize_path = os.path.join(synth_mcep_files, name)
    # load MCEP vectors
    ref_vec = np.load(filename)
    ref_frame_no = len(ref_vec)
    synth_vec = np.load(synthesize_path)

    # dynamic time warping using librosa
    min_cost, _ = librosa.sequence.dtw(ref_vec[:, 1:].T, synth_vec[:, 1:].T, metric=cost_function)
    min_cost_tot += np.mean(min_cost)
    frames_tot += ref_frame_no
  mean_mcd = min_cost_tot / frames_tot
  return mean_mcd, frames_tot

# Paths to target reference and converted synthesised wavs
names = ["baseline_fastspeech2", "fastspeech2_diffusion", "fastspeech2_diffusion_Style"]
for name in names:
  wave_groundtruth = glob.glob(f'./audio/synthesis_audio_interspeech2023_mcd/{name}/groundtruth/*')
  wave_synthesis = glob.glob(f'./audio/synthesis_audio_interspeech2023_mcd/{name}/synthesis/*')
  wave_groundtruth_np = f"./audio/synthesis_audio_interspeech2023_mcd/{name}_np/groundtruth/"
  wave_synthesis_np = f"./audio/synthesis_audio_interspeech2023_mcd/{name}_np/synthesis/"

  for wav in wave_groundtruth:
      wav2mcep_numpy(wav, wave_groundtruth_np, fft_size=fft_size, mcep_size=mcep_size)
  for wav in wave_synthesis:
      wav2mcep_numpy(wav, wave_synthesis_np, fft_size=fft_size, mcep_size=mcep_size)

  cost_function = log_spec_dB_dist
  mcd, frames_used = average_mcd(wave_groundtruth_np, wave_synthesis_np, cost_function)
  print(f'{name}: MCD = {mcd} dB, calculated over a total of {frames_used} frames')