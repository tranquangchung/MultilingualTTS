import glob
import librosa
import shutil
import os
import pdb

time = 0
limit_time = 960 # second

destination = "/data/raw/speech/S_synthesize/Multi_Lang/VN/train_adaptation/Adaptation_paper/lien_16m"
if not os.path.exists(destination):
    os.makedirs(destination)

files = glob.glob("/data/raw/speech/S_synthesize/Multi_Lang/VN/train_clean_VN_processed_22050_normalization_denoiser/0015_lien/*.txt")
for f in files:
    wav_file = f.replace("txt", "wav")
    print(f)
    print(wav_file)
    y, sr = librosa.load(wav_file)
    duration = librosa.get_duration(y)
    if duration < 4:
        continue
    print(duration)
    time += float(duration)
    shutil.copy2(f, destination)
    shutil.copy2(wav_file, destination)
    if time > limit_time:
        break
print(time)
