import glob
import os
import random
import json
import pdb

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

path = "/home/ldap-users/s2220411/Code/Multilingual_Data_Training/data_training/"
selective_sentences_path = "/home/ldap-users/s2220411/Code/Multilingual_Data_Training/selective_sentencesV3"
languages = os.listdir(path)
chinese_dictionary, korean_dictionary = load_textGT_chinese_and_korean()


# languages = ["japanese"]
for language in languages:
  with open(os.path.join(selective_sentences_path, language+".json"), "w", encoding='utf-8') as fout:
    speakers = os.listdir(os.path.join(path, language))
    text_file = []
    for speaker in speakers:
      text_file += glob.glob(os.path.join(path, language, speaker, "*.txt"))
      if len(text_file) >= 5000: break
    random.shuffle(text_file)
    index = 0
    list_data = []
    for text_path in text_file:
      filename = text_path.split("/")[-1].replace(".txt", "")
      with open(text_path, "r") as f_read:
        text = f_read.read()
        texttmp = text.replace(",","").replace(".","")
        numberword = len(texttmp.split())
        if (numberword < 8 or numberword > 16) and language not in ["japanese"]: continue # vi tiếng nhật dính liền với nhau
        if language == "vietnamese":
          if len(text.split()) >= 5:
            dictionary_representation = {"original_text": text,"text": text,"filename": filename,}
            list_data.append(dictionary_representation)
            index += 1
        elif language == "chinese":
          original_text = chinese_dictionary[filename]
          dictionary_representation = {"original_text": original_text,"text": text,"filename": filename,}
          list_data.append(dictionary_representation)
          index += 1
        elif language == "korean":
          original_text = korean_dictionary[filename]
          dictionary_representation = {"original_text": original_text,"text": text,"filename": filename,}
          list_data.append(dictionary_representation)
          index += 1
        else:
          dictionary_representation = {"original_text": text, "text": text, "filename": filename,}
          list_data.append(dictionary_representation)
          index += 1
      if index >= 500: break
    for item in list_data:
      json.dump(item, fout, ensure_ascii=False)
      fout.write("\n")
    print(language, len(list_data))