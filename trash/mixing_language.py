import os
import sys
sys.path.append("/home/chungtran/Code/C_tutorial/L2/.TTS/FastSpeech2")

import re
import argparse
from string import punctuation
import json
import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from g2p_en import G2p
from pypinyin import pinyin, Style
import pyopenjtalk
import librosa
from scipy.io.wavfile import write
import pdb

from text import text_to_sequence, text_to_sequence_mfa, load_lexicon_phone_mfa
from text import _symbols_to_sequence

with open("/data/processed/speech/tts/wav22050_TiengViet_2_speaker_MCV_150s_LJSpeech/languages.json") as f:
    language_map = json.load(f)

def add_prefix2phone(phone, lang):
    prefix = ""
    _silences = ["sp", "spn", "sil"]
    if lang == "Vie":
        prefix = "@vn_"
    elif lang == "Eng":
        prefix = "@eng_"
    elif lang == "Japan":
        # prefix = "jp_"
        prefix = ""
    elif lang == "Chi":
        prefix = "@cn_"
    prefix_phone = []
    for p in phone:
        if p not in _silences:
            prefix_phone.append(prefix+p)
        else:
            prefix_phone.append(p)
    return prefix_phone

def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon

lexicon_eng = read_lexicon("/home/chungtran/Code/C_tutorial/L2/.TTS/FastSpeech2/lexicon/librispeech-lexicon.txt")
lexicon_vie = load_lexicon_phone_mfa("/home/chungtran/Code/C_tutorial/L2/.TTS/montreal-forced-aligner_vn/Vietnamese_Lexicon/lexicon.txt.espeak.txt")
g2p = G2p()
def get_phone_and_lang():
    # text = "Find the latest breaking news and information on the top stories tìm tin tức nóng hổi mới nhất và thông tin về những tin bài hàng đầu"
    # language = ["Eng"] * 11 + ["Vie"] *16
    # text = "digital money electronic money electronic currency cyber cash tiền thuật toán tiền điện tử tiền mã hoá"
    # language =["Eng"] * 8 + ["Vie"] * 9
    text = "Golfer người Anh Matthew Fitzpatrick"
    language = ["Eng", "Vie", "Vie", "Eng", "Eng"]
    # language = ["Vie", "Vie", "Vie", "Vie", "Vie", "Vie", "Vie", "Eng", "Eng"]
    # text = "Oscar Otte cũng thua chung kết giải Stuttgart Open tuần trước dưới tay Matteo Berrettini "
    # language = ["Eng", "Eng", "Vie", "Vie", "Vie", "Vie", "Vie", "Eng", "Eng", "Vie", "Vie", "Vie", "Vie", "Eng", "Eng"]
    phoneme = []
    langs = []
    for t, l in zip(text.split(), language):
        if l == "Vie":
            lang_tmp = language_map[l]
            phoneme_tmp = lexicon_vie[t.lower()].split()
        elif l == "Eng":
            lang_tmp = language_map[l]
            if t.lower() in lexicon_eng:
                phoneme_tmp= lexicon_eng[t.lower()]
            else:
                phoneme_tmp = list(filter(lambda p: p != " ", g2p(t.lower())))
        phoneme += add_prefix2phone(phoneme_tmp, l)
        langs += [lang_tmp+1]*len(phoneme_tmp)
    symbol = _symbols_to_sequence(phoneme)
    print(langs)
    print(symbol)
    return symbol, langs
    # print(symbol)
    # print(len(symbol))
    # print(langs)
    # print(len(langs))
