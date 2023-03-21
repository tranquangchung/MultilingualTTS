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
    wav = vocoder_infer(postnet_output, vocoder, model_config, preprocess_config)
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
    write(path_orginal_audio, sr, wav_audio.astype(np.int16))

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
    return path_file_tmp, path_orginal_audio


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["batch", "single"],
        required=True,
        help="Synthesize a whole dataset or a single sentence",
    )
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
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
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

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # Get model
    ###########################################################################
    model = get_model_fastSpeech2_StyleEncoder_MultiLanguage_1(args, configs, device, train=False)
    model_name = "baseline_fastspeech2"

    # model = get_model_fastSpeech2_StyleEncoder_MultiLanguage_Difffusion1(args, configs, device, train=False)
    # model_name = "fastspeech2_diffusion"

    # model = get_model_fastSpeech2_StyleEncoder_MultiLanguage_Difffusion_Style1(args, configs, device, train=False)
    # model_name = "fastspeech2_diffusion_Style"
    ###########################################################################

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("pytorch_total_params", pytorch_total_params)
    print("pytorch_total_params_trainable", pytorch_total_params_trainable)
    # Load vocoder
    vocoder = get_vocoder(configs, device)

    with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "languages.json")) as f:
        language_map = json.load(f)
    # Preprocess texts
    # raw_path = preprocess_config['path']['raw_path']
    raw_path = "/home/ldap-users/s2220411/Code/Multilingual_Data_Training/data_training"
    language = ["chinese", "dutch", "english", "french", "german",
                "indonesian", "italian", "japanese", "korean",
                "polish", "portuguese", "russian", "spanish", "vietnamese"]
    Lang_Text = {}
    for lang in language:
        with open(f'/home/ldap-users/s2220411/Code/Multilingual_Data_Training/selective_sentences/{lang}.json') as fin:
            texts = fin.readlines()
            Lang_Text[lang] = texts
    kwargs = {"use_pinyin": True}
    corpus_path = "/home/ldap-users/s2220411/Code/FastSpeech2_multilingual/visualization/systhesis_audio_interspeech2023"
    target_duration = [3, 5, 10, 20]
    for target in target_duration:
        with open("./duration/gt_5secondV5.json", "r") as fin:
            lines = fin.readlines()
            for line in lines:
                line = json.loads(line)
                lang = line["language"]; speaker = line["speaker"]; file_name = line["filename"]; duration = line["duration"]
                target_text = line["text"]; length_original_target_text = line["numberword"]; original_target_text = line["original_text"]
                print(model_name, " target: ", target)
                print(line)
                path_file_target_duration, path_orginal_audio = processing_duration(raw_path, line, target=target)

                # path_file = os.path.join(raw_path, lang, speaker, file_name + ".wav")
                # processing duration audio
                # extract style speaker base on target duration
                style_vector = extract_style(model, path_file_target_duration)

                for lang_i in language:
                    random_number = random.randint(0, len(Lang_Text[lang_i])-1)
                    item_text = json.loads(Lang_Text[lang_i][random_number])
                    text_lang = item_text["text"]
                    original_text = item_text["original_text"]
                    text_lang = text_lang.translate(str.maketrans('', '', string.punctuation))
                    text_lang = text_lang.replace("—", "").replace('“', "").replace('”', "").replace("–", "")

                    processed_text = np.array([processes_all_languages(text_lang.strip(), lang_i, preprocess_config, **kwargs)])
                    text_lens = np.array([len(processed_text[0])])
                    lang_id = language_map[lang_i]
                    batch = [lang_id, processed_text, text_lens, max(text_lens)]
                    # batch = ["", "", lang_id, "", processed_text, text_lens, max(text_lens)]

                    ##########
                    control_values = args.pitch_control, args.energy_control, args.duration_control
                    wav_synthesis = synthesize(model, configs, vocoder, batch, control_values, style_vector)

                    # for visualize
                    os.makedirs(f"{corpus_path}/{model_name}/{target}_second/visualization/{lang}/{speaker}", exist_ok=True)
                    name_visualize = f"{corpus_path}/{model_name}/{target}_second/visualization/{lang}/{speaker}/{lang}__{speaker}__{file_name}__{lang_i}__{random_number}"
                    write(f"{name_visualize}.wav", 22050, wav_synthesis[0])

                    # # for evaluate MOS # divide by speaker
                    # os.makedirs(f"{corpus_path}/{model_name}/{target}_second/MOS/{lang}/{speaker}", exist_ok=True)
                    # name_mos = f"{corpus_path}/{model_name}/{target}_second/MOS/{lang}/{speaker}/{lang}__{speaker}__{file_name}__{lang_i}__{random_number}"
                    # write(f"{name_mos}.wav", 22050, wav_synthesis[0])
                    # with open(f"{name_mos}.txt", "w") as fout:
                    #     fout.write(original_text.strip())

                    # for evaluate MOS # divide by synthesized languages
                    os.makedirs(f"{corpus_path}/{model_name}/{target}_second/MOS_Lang/{lang_i}/", exist_ok=True)
                    name_mos = f"{corpus_path}/{model_name}/{target}_second/MOS_Lang/{lang_i}/{lang}__{speaker}__{file_name}__{lang_i}__{random_number}"
                    write(f"{name_mos}.wav", 22050, wav_synthesis[0])
                    with open(f"{name_mos}.txt", "w") as fout:
                        fout.write(original_text.strip())

                    # # for evaluate SIM with difference language and text
                    # os.makedirs(f"{corpus_path}/{model_name}/{target}_second/SIM_DiffLangText/{lang}/{speaker}", exist_ok=True)
                    # name_sim = f"{corpus_path}/{model_name}/{target}_second/SIM_DiffLangText/{lang}/{speaker}/{lang}__{speaker}__{file_name}__{lang_i}__{random_number}"
                    # write(f"{name_sim}.wav", 22050, wav_synthesis[0])
                    # # check file audio is already exist
                    # name_sim_groundtruth = f"{corpus_path}/{model_name}/{target}_second/SIM_DiffLangText/{lang}/{speaker}/{lang}__{speaker}__{file_name}__{lang_i}__{random_number}__groundtruth.wav"
                    # if not os.path.exists(name_sim_groundtruth):
                    #     shutil.copy2(path_orginal_audio, name_sim_groundtruth)


                # ### for evaluate SIM with same language and text
                # control_values = args.pitch_control, args.energy_control, args.duration_control
                #
                # target_text = target_text.translate(str.maketrans('', '', string.punctuation))
                # target_text = target_text.replace("—", "").replace('“', "").replace('”', "").replace("–", "")
                #
                # processed_text = np.array(
                #     [processes_all_languages(target_text.strip(), lang, preprocess_config, **kwargs)])
                # text_lens = np.array([len(processed_text[0])])
                # lang_id = language_map[lang]
                # batch = [lang_id, processed_text, text_lens, max(text_lens)]
                # wav_synthesis = synthesize(model, configs, vocoder, batch, control_values, style_vector)
                #
                # os.makedirs(f"{corpus_path}/{model_name}/{target}_second/SIM_SameLangText/{lang}/{speaker}", exist_ok=True)
                # name_sim = f"{corpus_path}/{model_name}/{target}_second/SIM_SameLangText/{lang}/{speaker}/{lang}__{speaker}__{file_name}__{lang}"
                # write(f"{name_sim}.wav", 22050, wav_synthesis[0])
                # # check file audio is already exist
                # name_sim_groundtruth = f"{corpus_path}/{model_name}/{target}_second/SIM_SameLangText/{lang}/{speaker}/{lang}__{speaker}__{file_name}__{lang}__groundtruth.wav"
                # if not os.path.exists(name_sim_groundtruth):
                #     shutil.copy2(path_orginal_audio, name_sim_groundtruth)
                #
                # # remove file target duration
                # os.remove(path_file_target_duration)