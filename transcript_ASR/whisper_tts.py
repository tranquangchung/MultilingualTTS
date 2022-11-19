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
path_dir = "/home/ldap-users/s2220411/Code/FastSpeech2_multilingual/visualization/wav_outputV2"
for lang in languages:
  speakers = os.listdir(os.path.join(path_dir, lang))
  for speaker in speakers:
    audio_files = glob.glob(os.path.join(path_dir, lang, speaker, "*"))
    for index, path in enumerate(audio_files):
      o_lang, o_speaker, o_filename, n_lang, id_text = path.split("/")[-1].split("__")
      n_lang = n_lang.replace("_", "") # some cases it append _
      id_text = int(id_text.replace(".wav", ""))
      groundtruth_langs.append(n_lang)
      grountruhth_text = Lang_Text[n_lang][id_text].strip()
      groundtruth_texts.append(grountruhth_text)

      #### whisper ####
      audio = whisper.load_audio(path)
      audio = whisper.pad_or_trim(audio)

      # make log-Mel spectrogram and move to the same device as the model
      mel = whisper.log_mel_spectrogram(audio).to(model.device)

      # detect the spoken language
      _, probs = model.detect_language(mel)
      predict_lang = max(probs, key=probs.get)

      predict_langs.append(Lang_Dict.get(predict_lang, predict_lang))

      # decode the audio
      options = whisper.DecodingOptions()
      result = whisper.decode(model, mel, options)
      predict_text = result.text
      predict_texts.append(predict_text)
      item = {
        "text_gt": grountruhth_text,
        "lang_gt": n_lang,
        "text_pd": predict_text,
        "lang_pd": predict_lang
      }
      print("*"*20)
      print(item)
      Processing_List.append(item)

with open('whisper_asr.json', 'w', encoding='utf-8') as f:
  json.dump(Processing_List, f, ensure_ascii=False, indent=4)

# correct = (np.array(predict_langs) == np.array(groundtruth_langs))
# accuracy = correct.sum() / correct.size
# print("Acc: ", accuracy*100)
#
# wer_error = wer(groundtruth_texts, predict_texts)
# cer_error = cer(groundtruth_texts, predict_texts)
# wil_error = wil(groundtruth_texts, predict_texts)
# wip_error = wip(groundtruth_texts, predict_texts)
# mer_error = mer(groundtruth_texts, predict_texts)
# print("wer_error: ", wer_error)
# print("cer_error: ", cer_error)
# print("wil_error: ", wil_error)
# print("wip_error: ", wip_error)
# print("wip_error:", mer_error)