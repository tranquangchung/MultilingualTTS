import re
import argparse
import os
import json
import torch
import yaml
import numpy as np

from utils.model import get_vocoder, get_model_fastSpeech2_StyleEncoder_MultiLanguage_Difffusion1
from utils.tool_lexicon import processes_all_languages
from utils.model import vocoder_infer
from utils.tool_audio import preprocess_audio
import audio as Audio
import librosa
from scipy.io.wavfile import write
import pdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def synthesize(model, configs, vocoder, batch, control_values, args):
#     preprocess_config, model_config, train_config = configs
#     pitch_control, energy_control, duration_control = control_values
#     _sftf = Audio.stft.TacotronSTFT(
#         preprocess_config["preprocessing"]["stft"]["filter_length"],
#         preprocess_config["preprocessing"]["stft"]["hop_length"],
#         preprocess_config["preprocessing"]["stft"]["win_length"],
#         preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
#         preprocess_config["preprocessing"]["audio"]["sampling_rate"],
#         preprocess_config["preprocessing"]["mel"]["mel_fmin"],
#         preprocess_config["preprocessing"]["mel"]["mel_fmax"],
#     )
#
#     wav, _ = librosa.load(args.ref_audio, sr=22050)
#     ref_mel_spectrogram, _ = Audio.tools.get_mel_from_wav(wav, _sftf)
#     ref_mel = torch.from_numpy(ref_mel_spectrogram).transpose(0,1).unsqueeze(0).to(device=device)
#
#     # Extract style vector
#     style_vector = model.get_style_vector(ref_mel)
#     # Forward
#     src = torch.from_numpy(batch[4]).to(device=device)
#     src_len = torch.from_numpy(batch[5]).to(device=device)
#     language = torch.tensor([batch[2]]).to(device=device)
#     max_src_len = batch[6]
#     output, postnet_output = model.inference(style_vector, src, language, src_len=src_len,
#                                              max_src_len=max_src_len, p_control=pitch_control,
#                                              e_control=energy_control, d_control=duration_control)
#     postnet_output = postnet_output.transpose(1, 2)
#     wav = vocoder_infer(postnet_output, vocoder, model_config, preprocess_config)
#     write("./wav_output/{0}_{1}_Diffusion.wav".format(args.name, args.language), 22050, wav[0])

def synthesize1(model, configs, vocoder, batch, control_values, args):
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
    wav = wav / max(abs(wav)) # normalize to 0-1
    ref_mel_spectrogram, _ = Audio.tools.get_mel_from_wav(wav, _sftf)
    ref_mel = torch.from_numpy(ref_mel_spectrogram).transpose(0,1).unsqueeze(0).to(device=device)
    # Extract style vector
    style_vector = model.get_style_vector(ref_mel)
    pdb.set_trace()
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
        default=1,
        help="control the speed of the whole utterance, larger value for slower speaking rate",
    )
    parser.add_argument("--ref_audio", type=str, required=True,
        help="path to an reference speech audio sample")
    parser.add_argument(
        "--model",
        type=str,
        choices=["naive", "aux", "shallow"],
        required=True,
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
    # model = get_model(args, configs, device, train=False)
    model = get_model_fastSpeech2_StyleEncoder_MultiLanguage_Difffusion1(args, configs, device, train=False)
    print(model)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("pytorch_total_params", pytorch_total_params)
    print("pytorch_total_params_trainable", pytorch_total_params_trainable)
    # Load vocoder
    vocoder = get_vocoder(configs, device)

    with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "languages.json")) as f:
        language_map = json.load(f)
    # Preprocess texts

    language = ["chinese", "dutch", "english", "french", "german",
                "indonesian", "italian", "japanese", "korean",
                "polish", "portuguese", "russian", "spanish", "vietnamese"]
    texts = [
        "這是日本先進科學技術研究所研究團隊開發的系統",
        "dit is het systeem dat is ontwikkeld door het onderzoeksteam van het Japanse geavanceerde instituut voor wetenschap en technologie",
        "this is the system developed by the research team of the japan advanced institute of science and technology",
        "c'est le système développé par l'équipe de recherche de l'institut supérieur des sciences et technologies du japon",
        "dieses system wurde vom forschungsteam des japan advanced institute of science and technology entwickelt",
        "ini adalah sistem yang dikembangkan oleh tim peneliti institut sains dan teknologi maju jepang",
        "questo è il sistema sviluppato dal gruppo di ricerca dell'istituto avanzato di scienza e tecnologia del Giappone",
        "これは北陸先端科学技術大学院大学の研究チームが開発したシステムです",
        "일본 첨단과학기술원 연구팀이 개발한 시스템입니다",
        "jest to system opracowany przez zespół badawczy japońskiego zaawansowanego instytutu nauki i technologii",
        "este é o sistema desenvolvido pela equipe de pesquisa do instituto avançado de ciência e tecnologia do japão",
        "это система, разработанная исследовательской группой японского передового института науки и техники",
        "este es el sistema desarrollado por el equipo de investigación del instituto avanzado de ciencia y tecnología de japón",
        "đây là hệ thống phát triển bởi đội nghiên cứu của viện khoa học và công nghệ tiên tiến nhật bản",
    ]

    for lang, text in zip(language, texts):
        args.language = lang
        args.text = text

        ids = raw_texts = [args.text[:100]]
        print("*"*20)
        print(args.text)
        texts = np.array([processes_all_languages(args.text, args.language, preprocess_config)])
        speakers = np.array([args.speaker_id])
        text_lens = np.array([len(texts[0])])
        lang_id = language_map[args.language]
        batch = [ids, raw_texts, lang_id, speakers, texts, text_lens, max(text_lens)]

        control_values = args.pitch_control, args.energy_control, args.duration_control
        # synthesize(model, configs, vocoder, batch, control_values, args)
        synthesize1(model, configs, vocoder, batch, control_values, args)
