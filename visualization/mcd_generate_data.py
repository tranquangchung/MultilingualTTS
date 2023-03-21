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
from utils.model import get_model_fastSpeech2_StyleEncoder_MultiLanguage_Difffusion1
from utils.model import get_model_fastSpeech2_StyleEncoder_MultiLanguage_Difffusion_Style1
from utils.model import get_model_fastSpeech2_StyleEncoder_MultiLanguage_Difffusion_Style_Language1
from utils.tools import to_device, synth_samples, pp_symbols, add_prefix2phone
from utils.model import vocoder_infer
from text import text_to_sequence
import pyopenjtalk
import audio as Audio
import librosa
from scipy.io.wavfile import write
import subprocess
import glob
import pdb
import random
from utils.tool_lexicon import processes_all_languages
import string
import shutil
import auditok
import uuid
import re
import random
from random import choice

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    ref_mel = torch.from_numpy(ref_mel_spectrogram).transpose(0,1).unsqueeze(0).to(device=device)

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
        return wav[0:(13*22050)]
    return wav

def processing_duration(raw_path, info, target=3, delta=0.5):
    max_wav_value = 32766
    # lang, speaker, file_name, duration = info.strip().split("|")

    lang = info["language"]; speaker = info["speaker"]; file_name = info["filename"]; duration = info["duration"]

    # duration = float(duration)
    path_file = os.path.join(raw_path, lang, speaker, file_name + ".wav")
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
    wav_audio_limit5s = wav_audio[0:(5*22050)]
    write(path_orginal_audio, sr, wav_audio_limit5s.astype(np.int16))
    ### take random other ground-truth
    path_mosgt = glob.glob(os.path.join(raw_path, lang, speaker, "*.wav"))
    random.shuffle(path_mosgt)
    path_mosgt = path_mosgt[0]
    ###
    if (target-delta) <= duration <= (target+delta):# around target +- delta
        pass # do nothing
    elif duration > target: # split to take enough data
        audio_regions = auditok.split(path_file_tmp, min_dur=2, max_dur=target, max_silence=0.3, energy_threshold=55)
        audio_list = [audio for audio in audio_regions]
        if len(audio_list) > 0:
            audio_list[0].save(path_file_tmp)
    elif duration <= target: # join audio to take enough data
        list_wavefiles = glob.glob(os.path.join(raw_path, lang, speaker, "*.wav"))
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
    return path_file_tmp, path_orginal_audio, path_mosgt

# def fake_audio(raw_path, random_speaker_item):
#     lang_r = random_speaker_item["language"]
#     speaker_r = random_speaker_item["speaker"]
#     path_orginal_audio_fake = glob.glob(os.path.join(raw_path, lang_r, speaker_r, "*.wav"))
#     random.shuffle(path_orginal_audio_fake)
#     wav_audio, sr = librosa.load(path_orginal_audio_fake[0])
#     wav_audio, _ = librosa.effects.trim(wav_audio)
#     wav_audio = wav_audio / max(abs(wav_audio)) * max_wav_value
#     prefix = str(uuid.uuid1())
#     path_orginal_audio = f"tmp/{prefix}_fake_groundtruth.wav"
#     write(path_orginal_audio, sr, wav_audio.astype(np.int16))
#     return path_orginal_audio

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
    args.restore_step = 480000
    args.preprocess_config = "../config/config_pcgpu/LibriTTS_StyleSpeech_multilingual/preprocess.yaml"
    args.model_config = "../config/config_pcgpu/LibriTTS_StyleSpeech_multilingual/model.yaml"
    args.train_config = "../config/config_pcgpu/LibriTTS_StyleSpeech_multilingual/train.yaml"
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)
    model = get_model_fastSpeech2_StyleEncoder_MultiLanguage_1(args, configs, device, train=False)
    model_name = "baseline_fastspeech2"
    yield [model, model_name, configs, args]

    ########################
    args.restore_step = 770000
    args.preprocess_config = "../config/config_pcgpu/LibriTTS_StyleSpeech_multilingual_diffusion_testV1/preprocess.yaml"
    args.model_config = "../config/config_pcgpu/LibriTTS_StyleSpeech_multilingual_diffusion_testV1/model.yaml"
    args.train_config = "../config/config_pcgpu/LibriTTS_StyleSpeech_multilingual_diffusion_testV1/train.yaml"
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)
    model = get_model_fastSpeech2_StyleEncoder_MultiLanguage_Difffusion1(args, configs, device, train=False)
    model_name = "fastspeech2_diffusion"
    yield [model, model_name, configs, args]

    # # # #########################
    args.restore_step = "20000_FreezeFS_3500_V2_1"
    args.preprocess_config = "../config/config_pcgpu/LibriTTS_StyleSpeech_multilingual_diffusion_style/preprocess.yaml"
    args.model_config = "../config/config_pcgpu/LibriTTS_StyleSpeech_multilingual_diffusion_style/model.yaml"
    args.train_config = "../config/config_pcgpu/LibriTTS_StyleSpeech_multilingual_diffusion_style/train.yaml"
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)
    model = get_model_fastSpeech2_StyleEncoder_MultiLanguage_Difffusion_Style1(args, configs, device, train=False)
    model_name = "fastspeech2_diffusion_Style"
    yield [model, model_name, configs, args]


if __name__ == "__main__":

    # Get model
    # Read Config
    raw_path = "/home/ldap-users/s2220411/Code/Multilingual_Data_Training/data_training"
    language = ["chinese", "english", "indonesian", "japanese", "korean", "vietnamese"]
    # language = ["vietnamese"]

    Lang_Text = {}
    for lang in language:
        with open(f'/home/ldap-users/s2220411/Code/Multilingual_Data_Training/selective_sentencesV4/{lang}.txt') as fin:
        # with open(f'/home/ldap-users/s2220411/Code/Multilingual_Data_Training/selective_sentences_VietNam/{lang}.txt') as fin:
            texts = fin.readlines()*4
            random.shuffle(texts)
            Lang_Text[lang] = texts
    kwargs = {"use_pinyin": True, "hangeul": True}
    corpus_path = "/home/ldap-users/s2220411/Code/FastSpeech2_multilingual/visualization/audio/synthesis_audio_interspeech2023_mcd"
    target_MOS_file = "gt_5secondV3_SIM_V2_edit.json"
    # target_MOS_file = "gt_5secondVietnam.json"

    total_line = []

    for [model, model_name, configs, args] in get_model():
        preprocess_config, model_config, train_config = configs
        vocoder = get_vocoder(configs, device)
        with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "languages.json")) as f:
            language_map = json.load(f)

        os.makedirs(f"{corpus_path}/{model_name}/synthesis", exist_ok=True)
        os.makedirs(f"{corpus_path}/{model_name}/groundtruth", exist_ok=True)

        with open(f"./duration/{target_MOS_file}", "r") as fin:
            target_speakers = fin.readlines()
        total_index = 0
        control_values = args.pitch_control, args.energy_control, args.duration_control
        for index, target_speaker in enumerate(target_speakers):
            target_speaker = json.loads(target_speaker)
            text_lang = target_speaker['text'].strip()
            n = _clean(text_lang)
            language = target_speaker['language']
            processed_text = np.array(
                [processes_all_languages(text_lang.strip(), language, preprocess_config, **kwargs)])
            print(processed_text)
            text_lens = np.array([len(processed_text[0])])
            lang_id = language_map[target_speaker['language']]
            batch = [lang_id, processed_text, text_lens, max(text_lens)]
            target_speaker_name = target_speaker["speaker"]
            print("-"*20)
            print("model_name: ", model_name)
            print("target_speaker_name:", target_speaker_name, "synthesized_lang:", lang_id)
            filename_speaker = target_speaker["filename"]
            path_orginal_audio = os.path.join(raw_path, language, target_speaker_name, filename_speaker+".wav")
            style_vector = extract_style(model, path_orginal_audio)
            print(filename_speaker, style_vector.max(), style_vector.min())
            wav_synthesis = synthesize(model, configs, vocoder, batch, control_values, style_vector)

            # for evaluate MOS # divide by synthesized languages
            filename2save = f"{lang_id}_{target_speaker_name}_{filename_speaker}_{total_index}"+".wav"
            filename_syn = os.path.join(corpus_path, model_name, "synthesis", filename2save)
            filename_gt = os.path.join(corpus_path, model_name, "groundtruth", filename2save)

            write(filename_syn, 22050, wav_synthesis)
            total_index += 1
            if not os.path.exists(filename_gt):
                shutil.copy2(path_orginal_audio, filename_gt)