import whisper
import glob
import os
import numpy as np
from jiwer import wer, cer, wil, wip, mer
import json
import pdb

model = whisper.load_model("medium")
predict_langs = []
predict_texts = []
groundtruth_langs = []
groundtruth_texts = []
#####
languages = ["chinese", "dutch", "english", "french", "german",
            "indonesian", "italian", "japanese", "korean",
            "polish", "portuguese", "russian", "spanish", "vietnamese"]
Lang_Text = {}
for lang in languages:
  with open(f'/home/ldap-users/s2220411/Code/FastSpeech2_multilingual/visualization/languages/{lang}.txt') as fin:
    texts = fin.readlines()
    Lang_Text[lang] = texts

#####
Lang_Dict = {
  "de": "german",
  "fr": "french",
  "vi": "vietnamese",
  "es": "spanish",
  "pl": "polish",
  "id": "indonesian",
  "pt": "portuguese",
  "zh": "chinese",
  "it": "italian",
  "en": "english",
  "nl": "dutch",
  "ru": "russian",
  "ja": "japanese",
  "ko": "korean",
}
Processing_List = []

ASR4Multilingual = {
  "baseline_fastspeech2": [],
  "fastspeech2_diffusion": [],
  "fastspeech2_diffusion_Style": [],
}

path_dir = "/home/ldap-users/s2220411/Code/FastSpeech2_multilingual/visualization/audio/synthesis_audio_interspeech2023_Unseen"
path_dir_import_file = os.path.join(path_dir, "importfile.txt")

options = whisper.DecodingOptions()
with open(path_dir_import_file) as fin:
  lines = fin.readlines()
  for line in lines:
    lang, model_tts, duration, grountruth_text, infer_wav, gt_wav = line.strip().split("|")
    if lang == "english":
      full_wav_path = os.path.join(path_dir, infer_wav)
      #### whisper ####
      audio = whisper.load_audio(full_wav_path)
      audio = whisper.pad_or_trim(audio)
      mel = whisper.log_mel_spectrogram(audio).to(model.device)
      # detect the spoken language
      _, probs = model.detect_language(mel)
      predict_lang = max(probs, key=probs.get)
      predict_lang = Lang_Dict.get(predict_lang, predict_lang)
      # decode the audio
      result = whisper.decode(model, mel, options)
      predict_text = result.text
      item = {
        "text_gt": grountruth_text,
        "lang_gt": lang,
        "text_pd": predict_text,
        "lang_pd": predict_lang
      }
      print(model_tts)
      print(item)
      ASR4Multilingual[model_tts].append(item)

with open('./results/multilingual.json', 'w', encoding='utf-8') as f:
  json.dump(ASR4Multilingual, f, ensure_ascii=False, indent=4)

# for lang in languages:
#   speakers = os.listdir(os.path.join(path_dir, lang))
#   for speaker in speakers:
#     audio_files = glob.glob(os.path.join(path_dir, lang, speaker, "*"))
#     for index, path in enumerate(audio_files):
#       o_lang, o_speaker, o_filename, n_lang, id_text = path.split("/")[-1].split("__")
#       n_lang = n_lang.replace("_", "") # some cases it append _
#       id_text = int(id_text.replace(".wav", ""))
#       groundtruth_langs.append(n_lang)
#       grountruhth_text = Lang_Text[n_lang][id_text].strip()
#       groundtruth_texts.append(grountruhth_text)
#
#       #### whisper ####
#       audio = whisper.load_audio(path)
#       audio = whisper.pad_or_trim(audio)
#
#       # make log-Mel spectrogram and move to the same device as the model
#       mel = whisper.log_mel_spectrogram(audio).to(model.device)
#
#       # detect the spoken language
#       _, probs = model.detect_language(mel)
#       predict_lang = max(probs, key=probs.get)
#
#       predict_langs.append(Lang_Dict.get(predict_lang, predict_lang))
#
#       # decode the audio
#       options = whisper.DecodingOptions()
#       result = whisper.decode(model, mel, options)
#       predict_text = result.text
#       predict_texts.append(predict_text)
#       item = {
#         "text_gt": grountruhth_text,
#         "lang_gt": n_lang,
#         "text_pd": predict_text,
#         "lang_pd": predict_lang
#       }
#       print("*"*20)
#       print(item)
#       Processing_List.append(item)
#
# with open('whisper_asr.json', 'w', encoding='utf-8') as f:
#   json.dump(Processing_List, f, ensure_ascii=False, indent=4)