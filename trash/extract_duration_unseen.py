import glob
import os
import librosa
import pdb
import random
import json

# language = "vietnamese"
path = "/home/ldap-users/Share/Corpora/Speech/multi/VTCK/wav48_silence_trimmed"
path_txt = "/home/ldap-users/Share/Corpora/Speech/multi/VTCK/txt"
path_wav_corpus = "/home/ldap-users/Share/Corpora/Speech/multi/VTCK/wav48_silence_trimmed"
speakers = os.listdir(path_txt)
number_speaker = 0
total_line = []
for speaker in speakers:
  file_txts = os.listdir(os.path.join(path_txt, speaker))
  for file_txt in file_txts:
    path_file_txt = os.path.join(path_txt, speaker, file_txt)
    with open(path_file_txt) as fread:
      text = fread.readlines()[0].strip()
      number_word = len(text.split())
      if number_word > 20:
        file_name = file_txt.split(".")[0]
        wav_path = os.path.join(path_wav_corpus, speaker, f"{file_name}_mic1.flac")
        if not os.path.exists(wav_path): continue
        wave_audio, sr = librosa.load(wav_path)
        wave_audio, index = librosa.effects.trim(wave_audio)
        duration = wave_audio.shape[0]/sr
        if duration > 7:
          dictionary_representation = {"language": "english", "speaker": speaker,
                                       "filename": f"{file_name}_mic1.flac", "duration": f"{duration}",
                                       "text": text, "numberword": number_word,
                                       "original_text": text}
          total_line.append(dictionary_representation)
          break
  number_speaker += 1
  if number_speaker == 5:
    break

with open("/home/ldap-users/s2220411/Code/FastSpeech2_multilingual/visualization/duration/Unseen_speaker.json", "w", encoding='utf-8') as fout:
  for item in total_line:
    json.dump(item, fout, ensure_ascii=False)
    fout.write("\n")


# languages = ["english"]
# total_line = []
# for language, speaker_package in zip(languages, [english_speaker]):
#   for speaker_index in speaker_package:
#     speaker, speaker_wav = speaker_index[0], speaker_index[1]
#     filename = os.path.join(path, language, speaker, speaker_wav)
#     wavefile = filename+".wavp"
#     textfile = filename+".txt"
#     y, sr = librosa.load(wavefile)
#     y, _ = librosa.effects.trim(y)
#     duration = y.shape[0] / sr
#     with open(textfile, "r") as fread:
#       text = fread.readlines()[0]
#       texttmp = text.replace(",", "").replace(".", "")
#       numberword = len(texttmp.split())
#
#       dictionary_representation = {"language": language, "speaker": speaker, "filename": speaker_wav, "duration": duration,
#                                    "text": text, "numberword": numberword, "original_text": original_text}
#       total_line.append(dictionary_representation)
#       print(dictionary_representation)

# with open("/home/ldap-users/s2220411/Code/FastSpeech2_multilingual/visualization/duration/gt_5secondV3_SIM.json", "w", encoding='utf-8') as fout:
#   for item in total_line:
#     json.dump(item, fout, ensure_ascii=False)
#     fout.write("\n")




