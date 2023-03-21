import sys
sys.path.append("..")
import re
import argparse
from string import punctuation
import os
import json
import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from g2p_en import G2p
from pypinyin import pinyin, Style
from korean_romanizer.romanizer import Romanizer

from utils.model import get_vocoder, get_model_fastSpeech2_StyleEncoder_MultiLanguage_1
from utils.model import get_model_fastSpeech2_StyleEncoder_MultiLanguage_Difffusion_Style1
from utils.tools import to_device, synth_samples, pp_symbols, add_prefix2phone
from utils.model import vocoder_infer
from text import text_to_sequence
from utils.tool_lexicon import processes_all_languages
import pyopenjtalk
import audio as Audio
import librosa
from scipy.io.wavfile import write
import subprocess
import glob
import shutil
import string
import shutil
import auditok
import uuid
import re
import random
from random import choice
import pdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# abbv_language = {
#     "dutch": "nl",
#     "french": "fr-fr",
#     "german": "de",
#     "indonesian": "id",
#     "italian": "it",
#     "korean": "ko",
#     "polish": "pl",
#     "portuguese": "pt",
#     "russian": "ru",
#     "spanish": "es",
#     "vietnamese": "vi",
# }

max_wav_value = 32766
haicham_re_remove = re.compile(r"([^\s]+):")
regex_multi_spaces = re.compile("\s{2,}")
regex_multi_commas = re.compile("(,\s){2,}")
regex_punc = re.compile("([\[\]\.,!?;:/])")
regex_clean = re.compile("[\"…“”'+\*\/]")

def _clean(text):

    for punc in list("():;"):
        text = text.replace(punc, " , ")
    for punc in list("!?-"):
        text = text.replace(punc, " . ")

    text = regex_multi_commas.sub(" , ", text)
    text = regex_multi_spaces.sub(" ", text)
    text = regex_clean.sub("", text)
    text = text.replace("-", " , ")
    text = text.replace("–", " , ")
    text = re.sub(r"\.+", ".", text)
    text = text.replace(",", " , ")
    text = text.replace(".", " . ")
    return text.strip()

# def openjtalk2julius(p3):
#     if p3 in ['A','I','U',"E", "O"]:
#         return "jp_" + p3.lower()
#     if p3 == 'cl':
#         return 'q'
#     if p3 == 'pau':
#         return 'sp'
#     if p3 == 'sil':
#         return 'sil'
#     return "jp_" + p3.lower()
#
# def read_lexicon(lex_path):
#     lexicon = {}
#     with open(lex_path) as f:
#         for line in f:
#             temp = re.split(r"\s+", line.strip("\n"))
#             word = temp[0]
#             phones = temp[1:]
#             if word.lower() not in lexicon:
#                 lexicon[word.lower()] = phones
#     return lexicon
#
# def load_lexicon_phone_mfa(filename):
#     lexicon_phone = dict()
#     with open(filename, encoding='utf-8') as f:
#         for index, line in enumerate(f):
#             lexicon, phone = line.strip().split("\t")
#             lexicon_phone[lexicon] = phone.strip()
#     return lexicon_phone
#
# def preprocess_english(text, preprocess_config):
#     text = text.rstrip(punctuation)
#     lexicon = read_lexicon("lexicon/dictionary/english-lexicon.txt")
#
#     g2p = G2p()
#     phones = []
#     words = re.split(r"([,;.\-\?\!\s+])", text)
#     for w in words:
#         if w.lower() in lexicon:
#             phones += lexicon[w.lower()]
#         else:
#             phones += list(filter(lambda p: p != " ", g2p(w)))
#     phones = add_prefix2phone(phones, lang="english")
#     phones = "{" + "}{".join(phones) + "}"
#     phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
#     phones = phones.replace("}{", " ")
#
#     sequence = np.array(
#         text_to_sequence(
#             phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
#         )
#     )
#
#     return np.array(sequence)
#
#
# def preprocess_mandarin(text, preprocess_config):
#     lexicon = read_lexicon("lexicon/dictionary/pinyin-lexicon-r.txt")
#
#     phones = []
#     pinyins = [
#         p[0]
#         for p in pinyin(
#             text, style=Style.TONE3, strict=False, neutral_tone_with_five=True
#         )
#     ]
#     for p in pinyins:
#         if p in lexicon:
#             phones += lexicon[p]
#         else:
#             phones.append("sp")
#
#     phones = add_prefix2phone(phones, lang="chinese")
#     phones = "{" + " ".join(phones) + "}"
#     sequence = np.array(
#         text_to_sequence(
#             phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
#         )
#     )
#
#     return np.array(sequence)
#
# def preprocess_japanese(text, preprocess_config):
#     fullcontext_labels = pyopenjtalk.extract_fullcontext(text)
#     phones, accents = pp_symbols(fullcontext_labels)
#     phones = [openjtalk2julius(p) for p in phones if p != '']
#     phones = add_prefix2phone(phones, lang="japanese")
#
#     phones = "{" + " ".join(phones) + "}"
#     sequence = np.array(
#         text_to_sequence(
#             phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
#         )
#     )
#
#     return np.array(sequence)
#
# def preprocess_language(text, preprocess_config, language):
#     lexicon = load_lexicon_phone_mfa(f"../lexicon/dictionary/{language}_espeak.txt")
#     words = text.split(" ")
#     phones = []
#     for word in words:
#         if word in lexicon:
#             phones += lexicon[word.lower()].split()
#         else:
#             command = "echo {0} | phonemize -l {1} -b espeak --language-switch remove-flags -p '-' -s '|' --strip".format(word, abbv_language[language])
#             phoneme = subprocess.getstatusoutput(command)[1].split("\n")[-1]
#             phoneme = phoneme.replace("-", " ")
#             phoneme = phoneme.strip().split()
#             phones += phoneme
#             print("Out of Vocabulary: ",word, phoneme)
#     phones = add_prefix2phone(phones, lang=language)
#     phones = "{" + " ".join(phones) + "}"
#     sequence = np.array(
#         text_to_sequence(
#             phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
#         )
#     )
#
#     return np.array(sequence)

def extract_style(model, ref_audio):
  _sftf = Audio.stft.TacotronSTFT(
    preprocess_config["preprocessing"]["stft"]["filter_length"],
    preprocess_config["preprocessing"]["stft"]["hop_length"],
    preprocess_config["preprocessing"]["stft"]["win_length"],
    preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
    preprocess_config["preprocessing"]["audio"]["sampling_rate"],
    preprocess_config["preprocessing"]["mel"]["mel_fmin"],
    preprocess_config["preprocessing"]["mel"]["mel_fmax"],
  )
  wav, _ = librosa.load(ref_audio, sr=22050)
  ref_mel_spectrogram, _ = Audio.tools.get_mel_from_wav(wav, _sftf)
  ref_mel = torch.from_numpy(ref_mel_spectrogram).transpose(0, 1).unsqueeze(0).to(device=device)

  # Extract style vector
  style_vector = model.get_style_vector(ref_mel)
  return style_vector

def synthesize(model, configs, vocoder, batch, control_values, style_vector):
  preprocess_config, model_config, train_config = configs
  pitch_control, energy_control, duration_control = control_values
  # Forward
  src = torch.from_numpy(batch[1]).to(device=device)
  src_len = torch.from_numpy(batch[2]).to(device=device)
  language = torch.tensor([batch[0]]).to(device=device)
  max_src_len = batch[3]
  output, postnet_output = model.inference(style_vector, src, language, src_len=src_len,
                                           max_src_len=max_src_len, p_control=pitch_control,
                                           e_control=energy_control, d_control=duration_control)
  postnet_output = postnet_output.transpose(1, 2)
  wav = vocoder_infer(postnet_output, vocoder, model_config, preprocess_config)[0]
  duration_synthesis = wav.shape[0] / 22050
  if duration_synthesis > 13:
    return wav[0:(13 * 22050)]
  return wav


def preprocess_audio(audio_file, _stft):
    wav, sample_rate = librosa.load(audio_file, sr=22050)
    mel_spectrogram, _ = Audio.tools.get_mel_from_wav(wav, _stft)
    return torch.from_numpy(mel_spectrogram).to(device=device)


def processing_duration(raw_path, speaker, target=3, delta=0.5):
  max_wav_value = 32766
  # duration = float(duration)
  path_file = os.path.join(raw_path, speaker, file_name + ".wav")
  wav_audio, sr = librosa.load(path_file)
  wav_audio, _ = librosa.effects.trim(wav_audio)
  wav_audio = wav_audio / max(abs(wav_audio)) * max_wav_value
  #### new duration ####
  duration = wav_audio.shape[0] / sr
  ######################
  prefix = str(uuid.uuid1())
  path_file_tmp = f"tmp/{file_name}_{prefix}.wav"
  write(path_file_tmp, sr, wav_audio.astype(np.int16))
  path_orginal_audio = f"tmp/{file_name}_{prefix}_groundtruth.wav"
  write(path_orginal_audio, sr, wav_audio.astype(np.int16))

  if (target - delta) <= duration <= (target + delta):  # around target +- delta
    pass  # do nothing
  elif duration > target:  # split to take enough data
    audio_regions = auditok.split(path_file_tmp, min_dur=2, max_dur=target, max_silence=0.3, energy_threshold=55)
    audio_list = [audio for audio in audio_regions]
    if len(audio_list) > 0:
      audio_list[0].save(path_file_tmp)
  elif duration <= target:  # join audio to take enough data
    list_wavefiles = glob.glob(os.path.join(raw_path, speaker, "*.wav"))
    random.shuffle(list_wavefiles)
    sil = np.zeros(int(22050 / 4), dtype=np.int16)
    for wavefile in list_wavefiles:
      wav_audio_tmp, _ = librosa.load(wavefile)
      wav_audio_tmp, _ = librosa.effects.trim(wav_audio_tmp)
      wav_audio_tmp = wav_audio_tmp / max(abs(wav_audio_tmp)) * max_wav_value
      wav_audio = np.concatenate((wav_audio, sil, wav_audio_tmp), 0)
      duration_append = wav_audio.shape[0] / sr
      if duration_append > target: break
    write(path_file_tmp, sr, wav_audio.astype(np.int16))
    audio_regions = auditok.split(path_file_tmp, min_dur=2, max_dur=target, max_silence=1.5, energy_threshold=55)
    audio_list = [audio for audio in audio_regions]
    if len(audio_list) > 0:
      audio_list[0].save(path_file_tmp)
  return path_file_tmp, path_orginal_audio

def get_model():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--source",
    type=str,
    default=None,
    help="path to a source file with format like train.txt and val.txt, for batch mode only",
  )
  parser.add_argument(
    "--text",
    type=str,
    default=None,
    help="raw text to synthesize, for single-sentence mode only",
  )
  parser.add_argument(
    "--name",
    type=str,
    default=None,
    help="name to save",
  )
  parser.add_argument(
    "--language",
    type=str,
    default=None,
    help="language to synthesis",
  )
  parser.add_argument(
    "--speaker_id",
    type=int,
    default=0,
    help="speaker ID for multi-speaker synthesis, for single-sentence mode only",
  )
  parser.add_argument(
    "--pitch_control",
    type=float,
    default=1.0,
    help="control the pitch of the whole utterance, larger value for higher pitch",
  )
  parser.add_argument(
    "--energy_control",
    type=float,
    default=1.0,
    help="control the energy of the whole utterance, larger value for larger volume",
  )
  parser.add_argument(
    "--duration_control",
    type=float,
    default=1.25,
    help="control the speed of the whole utterance, larger value for slower speaking rate",
  )
  parser.add_argument(
    "--model",
    type=str,
    choices=["naive", "aux", "shallow"],
    required=False,
    default="shallow",
    help="training model type",
  )
  args = parser.parse_args()

  ###########################################################################
  # args.restore_step = "10000_FS23500"
  # args.preprocess_config = "../config/config_pcgpu/LibriTTS_StyleSpeech_multilingual/preprocess.yaml"
  # args.model_config = "../config/config_pcgpu/LibriTTS_StyleSpeech_multilingual/model.yaml"
  # args.train_config = "../config/config_pcgpu/LibriTTS_StyleSpeech_multilingual/train.yaml"
  # preprocess_config = yaml.load(
  #   open(args.preprocess_config, "r"), Loader=yaml.FullLoader
  # )
  # model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
  # train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
  # configs = (preprocess_config, model_config, train_config)
  # model = get_model_fastSpeech2_StyleEncoder_MultiLanguage_1(args, configs, device, train=False)
  # model_name = "baseline_fastspeech2"
  # yield [model, model_name, configs, args]

  # # # ###########################################################################
  args.restore_step = 220000
  args.preprocess_config = "../config/config_pcgpu/LibriTTS_StyleSpeech_multilingual_diffusion_style_VN/preprocess.yaml"
  args.model_config = "../config/config_pcgpu/LibriTTS_StyleSpeech_multilingual_diffusion_style_VN/model.yaml"
  args.train_config = "../config/config_pcgpu/LibriTTS_StyleSpeech_multilingual_diffusion_style_VN/train.yaml"
  preprocess_config = yaml.load(
      open(args.preprocess_config, "r"), Loader=yaml.FullLoader
  )
  model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
  train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
  configs = (preprocess_config, model_config, train_config)
  model = get_model_fastSpeech2_StyleEncoder_MultiLanguage_Difffusion_Style1(args, configs, device, train=False)
  model_name = "vietname_model"
  yield [model, model_name, configs, args]


if __name__ == "__main__":
  path_folder = "/home/ldap-users/s2220411/Code/Multilingual_Data_Training/data_training/vietnamese"
  # speakers = os.listdir(path_folder)[:20]
  speakers = ["0000_ducnguyentrung", "0001_dungnguyentuan", "0002_nguyenthanhhai", "0005_thanhletien", "0012_hjeu",
             "0014_hongnhung", "0015_lien", "0016_minhnguyet", "0035_lethithuygiang", "0036_nguyenminhtrang"]
  # speaker_f = ["0014_hongnhung", "0015_lien", "0016_minhnguyet", "0035_lethithuygiang", "0036_nguyenminhtrang"]

  for [model, model_name, configs, args] in get_model():
    preprocess_config, model_config, train_config = configs
    vocoder = get_vocoder(configs, device)
    with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "languages.json")) as f:
        language_map = json.load(f)

    control_values = args.pitch_control, args.energy_control, args.duration_control
    args.language = "vietnamese"
    corpus_path = "/home/ldap-users/s2220411/Code/FastSpeech2_multilingual/visualization/audio/TSNE-PNP"
    kwargs = {"use_pinyin": True}
    for index, speaker in enumerate(speakers):
        wavs = glob.glob(os.path.join(path_folder, speaker, "*.wav"))
        count_file = 0
        for index_w, wav in enumerate(wavs):
            file_name = wav.split("/")[-1].split(".")[0]
            text_file = os.path.join(path_folder, speaker, file_name+".txt")
            with open(text_file, "r") as f:
                text = f.readlines()[0]
                text = text.replace("<spoken_noise>","")
                len_text = len(text.split())
                if len_text < 7: continue
                count_file += 1
                if count_file >= 11: break
            text = _clean(text)
            print(text)
            processed_text = np.array(
              [processes_all_languages(text.strip(), args.language, preprocess_config, **kwargs)])
            text_lens = np.array([len(processed_text[0])])
            lang_id = language_map[args.language]
            batch = [lang_id, processed_text, text_lens, max(text_lens)]
            style_vector = extract_style(model, wav)
            wav_synthesis = synthesize(model, configs, vocoder, batch, control_values, style_vector)

            file_synthesize = os.path.join(corpus_path, model_name, speaker, f"{file_name}_synthesis.wav")
            file_groundtruth = os.path.join(corpus_path, model_name, speaker, f"{file_name}.wav")
            os.makedirs("{0}/{1}/{2}".format(corpus_path, model_name, speaker), exist_ok=True)
            write(file_synthesize, 22050, wav_synthesis)
            if not os.path.exists(file_groundtruth):
              shutil.copy2(wav, file_groundtruth)