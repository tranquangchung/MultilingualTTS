import glob
import librosa
import shutil
import os

limit_time = 6000 # second
source = "/home/chungtran/Code/C_tutorial/L2/.TTS/StyleSpeech/dataset/wav22050_TiengViet_some_speaker_lien_10m"
name_fs = os.listdir(source)

for name_f in name_fs:
    files = glob.glob(os.path.join(source, name_f, "*.txt"))
    time = 0
    for f in files:
        wav_file = f.replace("txt", "wav")
        y, sr = librosa.load(wav_file)
        duration = librosa.get_duration(y)
        time += float(duration)
    if time < limit_time:
        index = 0
        max_index = len(files)
        while True: 
            f = files[(index%max_index)]
            wav_file = f.replace("txt", "wav")
            y, sr = librosa.load(wav_file)
            duration = librosa.get_duration(y)
            time += float(duration)

            ### soft-link here
            f_sl = f.replace(".txt", f"_sl_{index}.txt")
            wav_file_sl = wav_file.replace(".wav", f"_sl_{index}.wav")
            print(f_sl)
            print(wav_file_sl)
            print(time)
            os.symlink(f, f_sl)
            os.symlink(wav_file, wav_file_sl)
            index += 1
            if time > limit_time:
                break


