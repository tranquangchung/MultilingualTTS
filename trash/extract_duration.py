import glob
import os
import librosa
import pdb
import random
import json

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

# language = "vietnamese"
path = "/root/Code/Multilingual_Data_Training/data_training"
# language = ["chinese", "english", "indonesian", "japanese", "korean", "vietnamese"]
language = ["vietnamese"]


chinese_dictionary, korean_dictionary = load_textGT_chinese_and_korean()

total_line = []
for lang in language:
  speakers = os.listdir(os.path.join(path, lang))
  random.shuffle(speakers)
  speakers = speakers[:3] # take only 5 speakers for mos # 30 for sim
  if lang == "vietnamese":
    speakers = ["0015_lien"]
    # speakers = speakers[:1]
    # speakers.append("0015_lien")
    # speakers.append("0012_hjeu")
    # if "0055_viettts" in speakers: speakers.remove("0055_viettts")
    # speakers = list(set(speakers))
  for speaker in speakers:
    wavefiles = glob.glob(os.path.join(path, lang, speaker, "*.wav"))
    index_file = 0
    for wavefile in wavefiles:
      y, sr = librosa.load(wavefile)
      y, _ = librosa.effects.trim(y)
      duration = y.shape[0]/sr
      # check number word in file:
      textfile = wavefile.replace(".wav",".txt")
      with open(textfile, "r") as fread:
        text = fread.readlines()[0]
        texttmp = text.replace(",","").replace(".","")
        numberword = len(texttmp.split())
        # print(text, "-", numberword, "-", duration)
        # if (numberword < 8 or numberword > 16) and lang != "japanese": continue
      if duration > 3:
        filename = wavefile.split("/")[-1].split(".")[0]
        original_text = text
        if lang == "chinese":
          original_text = chinese_dictionary[filename]
        if lang == "korean":
          original_text = korean_dictionary[filename]
        dictionary_representation = {"language": lang, "speaker": speaker, "filename": filename, "duration": duration,
          "text": text, "numberword": numberword, "original_text": original_text}
        print(dictionary_representation)
        # line = f"{lang}||{speaker}||{filename}||{duration}||{text}||{numberword}||{original_text}"
        # print(line)
        total_line.append(dictionary_representation)
        index_file += 1
        if index_file >= 20: break # 5 files for mos # 10 files for sim

with open("/home/ldap-users/s2220411/Code/FastSpeech2_multilingual/visualization/duration/gt_5secondVietnam.json", "w", encoding='utf-8') as fout:
  for item in total_line:
    json.dump(item, fout, ensure_ascii=False)
    fout.write("\n")
  # for line in total_line:
  #   fout.write(line+"\n")