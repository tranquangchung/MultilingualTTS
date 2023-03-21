import json
import glob
import os
import pdb
import shutil
import librosa
from scipy.io.wavfile import write
import numpy as np

def load_textGT_chinese_and_korean():
  #################################
  # load chinese sentences corpus #
  #################################
  chinese_dictionary = {}
  path_chinese = "/home/ldap-users/Share/Corpora/Speech/multi/Multi_Lang/china_tts/test/content.txt"
  with open(path_chinese, "r") as fread:
    lines = fread.readlines()
    for line in lines:
      filename, sentence = line.split("\t")
      chinese_dictionary[filename.replace(".wav", "")] = sentence.strip()
  path_chinese = "/home/ldap-users/Share/Corpora/Speech/multi/Multi_Lang/china_tts/train/content.txt"
  with open(path_chinese, "r") as fread:
    lines = fread.readlines()
    for line in lines:
      filename, sentence = line.split("\t")
      chinese_dictionary[filename.replace(".wav", "")] = sentence.strip()

  #################################
  # load korean sentences corpus #
  #################################
  korean_dictionary = {}
  path = "/home/ldap-users/Share/Corpora/Speech/multi/unzip_data"
  languages = "zeroth_korean"
  folders = ["train_data_01", "test_data_01"]
  for folder in folders:
    path_tmp = os.path.join(path, languages, folder, "003")
    speakers = os.listdir(path_tmp)
    for speaker in speakers:
      trans_txt = os.path.join(path, languages, folder, "003", speaker, f"{speaker}_003.trans.txt")
      with open(trans_txt) as f:
        lines = f.readlines()
        for line in lines:
          name_file, text = line.strip().split(" ", 1)
          korean_dictionary[name_file] = text.strip()
  return chinese_dictionary, korean_dictionary

raw_path = "/home/ldap-users/s2220411/Code/Multilingual_Data_Training/data_training"
language = ["chinese", "english", "indonesian", "japanese", "korean", "vietnamese"]

chinese_dictionary, korean_dictionary = load_textGT_chinese_and_korean()

target_MOS_file = "gt_5secondV3_SIM.json"
corpus_path = "/home/ldap-users/s2220411/Code/FastSpeech2_multilingual/visualization/synthesis_audio_interspeech2023_final_mos1"
os.makedirs(f"{corpus_path}", exist_ok=True)
total_line = []
### MOS GT ###
max_wav_value = 32766
with open(f"/home/ldap-users/s2220411/Code/FastSpeech2_multilingual/visualization/duration/{target_MOS_file}", "r") as fin:
  target_speakers = fin.readlines()
  for target_speaker in target_speakers:
    target_speaker = json.loads(target_speaker)
    lang = target_speaker["language"]
    speaker = target_speaker["speaker"]
    filename = target_speaker["filename"]
    path_mosgt = glob.glob(os.path.join(raw_path, lang, speaker, "*.wav"))
    index_speaker = 0
    file_audiogt = os.path.join(raw_path, lang, speaker, filename + ".wav")

    wav_audio_gt, sr = librosa.load(file_audiogt)
    wav_audio_gt, _ = librosa.effects.trim(wav_audio_gt)
    wav_audio_gt = wav_audio_gt / max(abs(wav_audio_gt)) * max_wav_value
    wav_audio_gt = wav_audio_gt[0:(5*22050)]

    for wav_path in path_mosgt:
      wav_audio, sr = librosa.load(wav_path)
      wav_audio, _ = librosa.effects.trim(wav_audio)
      wav_audio = wav_audio / max(abs(wav_audio)) * max_wav_value

      duration = wav_audio.shape[0] / sr
      text_path = wav_path.replace(".wav", ".txt")
      filename_mos = wav_path.split("/")[-1].replace(".wav", "")
      with open(text_path) as fread:
        original_text = fread.readlines()[0].strip()
      namemos = f"{lang}_{speaker}_{filename_mos}_{index_speaker}_mos.wav"
      namesim = f"{lang}_{speaker}_{filename_mos}_{index_speaker}_sim.wav"

      if lang == "chinese":
        original_text = chinese_dictionary[filename]
      if lang == "korean":
        original_text = korean_dictionary[filename]
      if duration < 10:
        # wav_audio = wav_audio[0:(5*22050)]
        index_speaker += 1
        text_name = f"{lang}|groundtruth|0|{original_text.strip()}|{namemos}|{namesim}"
        print(text_name)
        filemos = os.path.join(corpus_path, namemos)
        filesim = os.path.join(corpus_path, namesim)
        write(filemos, sr, wav_audio.astype(np.int16))
        write(filesim, sr, wav_audio_gt.astype(np.int16))
        # shutil.copy2(wav_path, filemos)
        # shutil.copy2(file_audiogt, filesim)
        total_line.append(text_name)
      if index_speaker == 6:
        break

filename_txt = os.path.join(corpus_path, "importfile.txt")
with open(filename_txt, "w") as fout:
    for line in total_line:
        fout.write(line.strip() + "\n")