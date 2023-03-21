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
language = ["chinese", "english", "indonesian", "japanese", "korean", "vietnamese"]
# language = ["vietnamese"]


chinese_dictionary, korean_dictionary = load_textGT_chinese_and_korean()

# total_line = []
# for lang in language:
#   speakers = os.listdir(os.path.join(path, lang))
#   random.shuffle(speakers)
#   speakers = speakers[:3] # take only 5 speakers for mos # 30 for sim
#   if lang == "vietnamese":
#     speakers = ["0015_lien", "0012_hjeu", "0013_aihoa"]
#     # speakers = speakers[:1]
#     # speakers.append("0015_lien")
#     # speakers.append("0012_hjeu")
#     # if "0055_viettts" in speakers: speakers.remove("0055_viettts")
#     # speakers = list(set(speakers))
#   for speaker in speakers:
#     wavefiles = glob.glob(os.path.join(path, lang, speaker, "*.wav"))
#     index_file = 0
#     for wavefile in wavefiles:
#       y, sr = librosa.load(wavefile)
#       y, _ = librosa.effects.trim(y)
#       duration = y.shape[0]/sr
#       # check number word in file:
#       textfile = wavefile.replace(".wav",".txt")
#       with open(textfile, "r") as fread:
#         text = fread.readlines()[0]
#         texttmp = text.replace(",","").replace(".","")
#         numberword = len(texttmp.split())
#         # print(text, "-", numberword, "-", duration)
#         if (numberword < 8 or numberword > 16) and lang != "japanese": continue
#       if duration > 3:
#         filename = wavefile.split("/")[-1].split(".")[0]
#         original_text = text
#         if lang == "chinese":
#           original_text = chinese_dictionary[filename]
#         if lang == "korean":
#           original_text = korean_dictionary[filename]
#         dictionary_representation = {"language": lang, "speaker": speaker, "filename": filename, "duration": duration,
#           "text": text, "numberword": numberword, "original_text": original_text}
#         print(dictionary_representation)
#         # line = f"{lang}||{speaker}||{filename}||{duration}||{text}||{numberword}||{original_text}"
#         # print(line)
#         total_line.append(dictionary_representation)
#         index_file += 1
#         if index_file >= 1: break # 5 files for mos # 10 files for sim
#
# with open("/home/ldap-users/s2220411/Code/FastSpeech2_multilingual/visualization/duration/gt_5secondV3_SIM.json", "w", encoding='utf-8') as fout:
#   for item in total_line:
#     json.dump(item, fout, ensure_ascii=False)
#     fout.write("\n")
#   # for line in total_line:
#   #   fout.write(line+"\n")

chinese_speaker = [["SSB0241", "SSB02410004"],
                   ["SSB0122", "SSB01220001"],
                   ["SSB1956", "SSB19560001"]]
indo_speaker = [["Ind001", "Ind001_F_B_C_news_0002"],
                ["Ind003", "Ind003_M_U_C_news_0068"],
                ["Ind013", "Ind013_M_U_C_news_0396"]]
korean_speaker = [["105", "105_003_2853"],
                  ["106", "106_003_0077"],
                  ["110", "110_003_0026"]]
vietnamese_speaker = [["0012_hjeu", "0012_hjeu_hjeu-10-0023"],
                      ["0015_lien", "0015_lien_lien-104-0028"],
                      ["0013_aihoa", "0013_aihoa_f112aa_AiHoa_0024_0100"]]
english_speaker = [["100", "100_121669_000004_000000"],
                   ["1028", "1028_133393_000002_000000"],
                   ["1066", "1066_103481_000013_000000"]]
japanese_speaker = [["jvs004", "jvs004_BASIC5000_0074"],
                    ["jvs007", "jvs007_BASIC5000_0543"],
                    ["jvs012", "jvs012_BASIC5000_0202"]]
languages = ["chinese", "indonesian", "korean", "vietnamese", "english", "japanese"]
total_line = []
for language, speaker_package in zip(languages, [chinese_speaker, indo_speaker, korean_speaker, vietnamese_speaker, english_speaker, japanese_speaker]):
  for speaker_index in speaker_package:
    speaker, speaker_wav = speaker_index[0], speaker_index[1]
    filename = os.path.join(path, language, speaker, speaker_wav)
    wavefile = filename+".wav"
    textfile = filename+".txt"
    y, sr = librosa.load(wavefile)
    y, _ = librosa.effects.trim(y)
    duration = y.shape[0] / sr
    with open(textfile, "r") as fread:
      text = fread.readlines()[0]
      texttmp = text.replace(",", "").replace(".", "")
      numberword = len(texttmp.split())

      if language == "chinese":
        original_text = chinese_dictionary[speaker_wav]
      if language == "korean":
        original_text = korean_dictionary[speaker_wav]
      dictionary_representation = {"language": language, "speaker": speaker, "filename": speaker_wav, "duration": duration,
                                   "text": text, "numberword": numberword, "original_text": original_text}
      total_line.append(dictionary_representation)
      print(dictionary_representation)

with open("/home/ldap-users/s2220411/Code/FastSpeech2_multilingual/visualization/duration/gt_5secondV3_SIM.json", "w", encoding='utf-8') as fout:
  for item in total_line:
    json.dump(item, fout, ensure_ascii=False)
    fout.write("\n")




