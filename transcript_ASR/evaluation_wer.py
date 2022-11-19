import sys
sys.path.append("..")
from jiwer import wer, cer, wil, wip, mer
import os
import json
import string
import time
from pypinyin import pinyin, Style
from pycnnum import num2cn
from korean_romanizer.romanizer import Romanizer
from utils.tools import add_prefix2phone, pp_symbols, openjtalk2julius
import pyopenjtalk
import re
import matplotlib.pyplot as plt
import matplotlib
import pdb



punctuation = """
–!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~
"""
ground_truth_d = {}
hypothesis_d = {}
languages = ["chinese", "dutch", "english", "french", "german",
            "indonesian", "italian", "japanese", "korean",
            "polish", "portuguese", "russian", "spanish", "vietnamese"]
for lang in languages:
  ground_truth_d[lang] = []
  hypothesis_d[lang] = []

Lang_Dict = {
  "de": "german",
  "fr": "french",
  "vi": "vietnamese",
  "es": "spanish",
  "pl": "polish",
  "id": "indonesian",
  "jw": "indonesian",
  "pt": "portuguese",
  "zh": "chinese",
  "it": "italian",
  "en": "english",
  "nl": "dutch",
  "ru": "russian",
  "ja": "japanese",
  "ko": "korean",
}

fin = open("whisper_asr.json", 'r')
data_whisper = json.load(fin)
ground_truth = []
hypothesis = []

lang_gts = []
lang_hyps = []

def draw_bar_char(data, languages, name):
  plt.clf()
  # Figure Size
  fig, ax = plt.subplots(figsize=(16, 9))
  # Horizontal Bar Plot
  ax.barh(languages, data)
  # Remove axes splines
  for s in ['top', 'bottom', 'left', 'right']:
    ax.spines[s].set_visible(False)

  # Remove x, y Ticks
  ax.xaxis.set_ticks_position('none')
  ax.yaxis.set_ticks_position('none')

  # Add x, y gridlines
  # ax.grid(b=True, color='grey',linestyle='-.', linewidth=0.5, alpha=0.2)

  # Show top values
  ax.invert_yaxis()

  # Add annotation to bars
  for i in ax.patches:
    plt.text(i.get_width()*1.002, i.get_y() + 0.5, str(round((i.get_width()), 3)),
             fontsize=10, color='black')
  # Add Plot Title
  ax.set_title("{0}".format(name.replace("_", " ")))
  plt.savefig(f"{name}.png", dpi=400)

def remove_punctuation(text):
  text = text.lower()
  text = text.translate(str.maketrans('', '', string.punctuation))
  text = text.replace("—", "").replace('“', "").replace('”', "").replace("–", "")
  text = text.replace("、", "").replace("。", "")
  return text

def processing_japanese(text):
  fullcontext_labels = pyopenjtalk.extract_fullcontext(text)
  phones, accents = pp_symbols(fullcontext_labels)
  phones = [openjtalk2julius(p, prefix="") for p in phones if p != '']
  return " ".join(phones).strip()

def extract_number(text):
  return re.findall(r'\b\d+\b', text)

def calculate_text_error(ground_truth, hypothesis):
  wer_error = wer(ground_truth, hypothesis)
  cer_error = cer(ground_truth, hypothesis)
  wil_error = wil(ground_truth, hypothesis)
  wip_error = wip(ground_truth, hypothesis)
  mer_error = mer(ground_truth, hypothesis)
  return round(wer_error, 3), round(cer_error, 3), round(wil_error, 3)\
    , round(wip_error, 3), round(mer_error,3)

for item in data_whisper:
  lang_gt = item["lang_gt"]
  lang_pd = Lang_Dict.get(item["lang_pd"], item["lang_pd"])
  text_gt = remove_punctuation(item["text_gt"])
  text_pd = remove_punctuation(item["text_pd"])
  if lang_pd == "chinese":
    pinyins = [p[0] for p in pinyin(text_pd, style=Style.TONE3, strict=False, neutral_tone_with_five=True)]
    text_pd = " ".join(pinyins).strip()
  if lang_pd == "korean":
    r = Romanizer(text_pd)
    text_pd = r.romanize()
  if lang_pd == "japanese":
    text_gt = processing_japanese(text_gt)
    text_pd = processing_japanese(text_pd)
  ground_truth.append(text_gt)
  hypothesis.append(text_pd)
  lang_gts.append(lang_gt)
  lang_hyps.append(lang_pd)
  if lang_pd == lang_gt:
    ground_truth_d[lang_gt].append(text_gt)
    hypothesis_d[lang_pd].append(text_pd)

wer_langs = []
cer_langs = []
wil_langs = []
wip_langs = []
mer_langs = []
for lang in languages:
  hypothesis_tmp = hypothesis_d[lang]
  ground_truth_tmp = ground_truth_d[lang]
  print("Lang: {0}, size: {1}".format(lang, len(hypothesis_tmp)))
  wer_error, cer_error, wil_error, wip_error, mer_error = calculate_text_error(hypothesis_tmp, ground_truth_tmp)
  print(wer_error, cer_error, wil_error, wip_error, mer_error)
  wer_langs.append(wer_error)
  cer_langs.append(cer_error)
  wil_langs.append(wil_error)
  wip_langs.append(wip_error)
  mer_langs.append(mer_error)

draw_bar_char(wer_langs, languages, "Word_Error_Rate")
draw_bar_char(cer_langs, languages, "Character_Error_Rate")
draw_bar_char(wil_langs, languages, "Word_Information_Lost")
draw_bar_char(wip_langs, languages, "Word_Information_Preserved")
draw_bar_char(mer_langs, languages, "Match_Error_Rate")