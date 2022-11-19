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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def synthesize(model, configs, vocoder, batch, control_values, args):
    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control = control_values
    _sftf = Audio.stft.TacotronSTFT(
        preprocess_config["preprocessing"]["stft"]["filter_length"],
        preprocess_config["preprocessing"]["stft"]["hop_length"],
        preprocess_config["preprocessing"]["stft"]["win_length"],
        preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        preprocess_config["preprocessing"]["audio"]["sampling_rate"],
        preprocess_config["preprocessing"]["mel"]["mel_fmin"],
        preprocess_config["preprocessing"]["mel"]["mel_fmax"],
    )

    wav, _ = librosa.load(args.ref_audio, sr=22050)
    ref_mel_spectrogram, _ = Audio.tools.get_mel_from_wav(wav, _sftf)
    ref_mel = torch.from_numpy(ref_mel_spectrogram).transpose(0,1).unsqueeze(0).to(device=device)

    # Extract style vector
    style_vector = model.get_style_vector(ref_mel)
    # Forward
    src = torch.from_numpy(batch[4]).to(device=device)
    src_len = torch.from_numpy(batch[5]).to(device=device)
    language = torch.tensor([batch[2]]).to(device=device)
    max_src_len = batch[6]
    output, postnet_output = model.inference(style_vector, src, language, src_len=src_len,
                                             max_src_len=max_src_len, p_control=pitch_control,
                                             e_control=energy_control, d_control=duration_control)
    postnet_output = postnet_output.transpose(1, 2)
    wav = vocoder_infer(postnet_output, vocoder, model_config, preprocess_config)
    write("./wav_output/{0}_{1}_hifigan.wav".format(args.name, args.language), 22050, wav[0])


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

    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # Get model
    # model = get_model(args, configs, device, train=False)
    model = get_model_fastSpeech2_StyleEncoder_MultiLanguage_1(args, configs, device, train=False)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("pytorch_total_params", pytorch_total_params)
    print("pytorch_total_params_trainable", pytorch_total_params_trainable)
    # Load vocoder
    vocoder = get_vocoder(configs, device)

    with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "languages.json")) as f:
        language_map = json.load(f)
    # Preprocess texts
    raw_path = preprocess_config['path']['raw_path']
    language = ["chinese", "dutch", "english", "french", "german",
                "indonesian", "italian", "japanese", "korean",
                "polish", "portuguese", "russian", "spanish", "vietnamese"]
    Lang_Text = {}
    for lang in language:
        with open(f'./languages/{lang}.txt') as fin:
            texts = fin.readlines()
            Lang_Text[lang] = texts

    with open("./duration/5_10second.txt") as fin:
        lines = fin.readlines()
        for line in lines:
            lang, speaker, file_name, duration = line.strip().split("|")
            path_file = os.path.join(raw_path, lang, speaker, file_name + ".wav")
            for lang_i in language:
                random_text = random.randint(0, 99)
                text_lang = Lang_Text[lang_i][random_text]
                kwargs = {
                    "use_pinyin": True,
                }
                pdb.set_trace()
                processed_text = np.array([processes_all_languages(text_lang, lang_i, preprocess_config, **kwargs)])
                print(speaker, lang_i, processed_text)

    # with open("./duration/15_20second.txt", "w") as fout:
    #     for lang in language:
    #         path_lang = os.path.join(raw_path, lang)
    #         speakers = os.listdir(path_lang)
    #         args.language = lang
    #         for speaker in speakers:
    #             path_speaker = os.path.join(path_lang, speaker)
    #             wavs = glob.glob(os.path.join(path_speaker, "*.wav"))
    #             total_speaker_index = 0
    #             total_wav_file = []
    #             for wav_path in wavs:
    #                 file_name = wav_path.split("/")[-1].split(".")[0]
    #                 wav, _ = librosa.load(wav_path)
    #                 duration = librosa.get_duration(wav)
    #                 ############ processing ###########
    #                 if 15 < duration < 20:
    #                     item = [lang, speaker, file_name, "%.2f"%duration]
    #                     print(item)
    #                     total_wav_file.append(item)
    #                 if len(total_wav_file) >= 5:
    #                     break
    #             if len(total_wav_file) >= 5:
    #                 for line in total_wav_file[:5]:
    #                     i_lang, i_speaker, i_filename, i_duration = lang, speaker, file_name, duration
    #                     text_line = "{0}|{1}|{2}|{3}\n".format(i_lang, i_speaker, i_filename, i_duration)
    #                     fout.write(text_line)
    #                 total_speaker_index += 1
    #                 if total_speaker_index == 5: break

            # for line in line_texts[:100]:
            #     fout.write(line+"\n")

    # pdb.set_trace()
    # texts = [
    #     "這是日本先進科學技術研究所研究團隊開發的系統",
    #     "dit is het systeem dat is ontwikkeld door het onderzoeksteam van het Japanse geavanceerde instituut voor wetenschap en technologie",
    #     "this is the system developed by the research team of the japan advanced institute of science and technology",
    #     "c'est le système développé par l'équipe de recherche de l'institut supérieur des sciences et technologies du japon",
    #     "dieses system wurde vom forschungsteam des japan advanced institute of science and technology entwickelt",
    #     "ini adalah sistem yang dikembangkan oleh tim peneliti institut sains dan teknologi maju jepang",
    #     "questo è il sistema sviluppato dal gruppo di ricerca dell'istituto avanzato di scienza e tecnologia del Giappone",
    #     "これは北陸先端科学技術大学院大学の研究チームが開発したシステムです",
    #     "일본 첨단과학기술원 연구팀이 개발한 시스템입니다",
    #     "jest to system opracowany przez zespół badawczy japońskiego zaawansowanego instytutu nauki i technologii",
    #     "este é o sistema desenvolvido pela equipe de pesquisa do instituto avançado de ciência e tecnologia do japão",
    #     "это система, разработанная исследовательской группой японского передового института науки и техники",
    #     "este es el sistema desarrollado por el equipo de investigación del instituto avanzado de ciencia y tecnología de japón",
    #     "đây là hệ thống phát triển bởi đội nghiên cứu của viện khoa học và công nghệ nhật bản",
    # ]
    #
    # for lang, text in zip(language, texts):
    #     args.language = lang
    #     args.text = text
    #
    #     ids = raw_texts = [args.text[:100]]
    #     print("*"*20)
    #     print(args.text)
    #     speakers = np.array([args.speaker_id])
    #     if args.language == "chinese":
    #         texts = np.array([preprocess_mandarin(args.text, preprocess_config)])
    #     elif args.language == "japanese":
    #         texts = np.array([preprocess_japanese(args.text, preprocess_config)])
    #     elif args.language == "korean":
    #         r = Romanizer(text)
    #         text_korean = r.romanize()
    #         texts = np.array([preprocess_language(text_korean, preprocess_config, args.language)])
    #     elif args.language == "english":
    #         texts = np.array([preprocess_english(args.text, preprocess_config)])
    #     elif args.language in ["dutch", "french", "german", "indonesian",
    #                            "polish", "portuguese", "russian", "spanish", "italian", "vietnamese"
    #                         ]:
    #         texts = np.array([preprocess_language(args.text, preprocess_config, args.language)])
    #     text_lens = np.array([len(texts[0])])
    #     lang_id = language_map[args.language]
    #     batch = [ids, raw_texts, lang_id, speakers, texts, text_lens, max(text_lens)]
    #
    #     control_values = args.pitch_control, args.energy_control, args.duration_control
    #     synthesize(model, configs, vocoder, batch, control_values, args)
