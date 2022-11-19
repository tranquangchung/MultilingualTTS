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

from utils.model import get_model, get_vocoder, get_model_fastSpeech2_StyleEncoder_MultiLanguage, get_model_styleencoder_multispeaker 
from utils.tools import to_device, synth_samples, pp_symbols, add_prefix2phone
from utils.model import vocoder_infer
from dataset import TextDataset, text_to_sequence_phone_vn_mfa
from text import text_to_sequence, text_to_sequence_mfa, load_lexicon_phone_mfa
import pyopenjtalk
import audio as Audio
import librosa
from scipy.io.wavfile import write
import pdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def openjtalk2julius(p3):
    if p3 in ['A','I','U',"E", "O"]:
        return "jp_" + p3.lower()
    if p3 == 'cl':
        return 'q'
    if p3 == 'pau':
        return 'sp'
    if p3 == 'sil':
        return 'sil'
    return "jp_" + p3.lower()

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


def preprocess_english(text, preprocess_config):
    text = text.rstrip(punctuation)
    # lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])
    lexicon = read_lexicon("/home/chungtran/Code/TTS/FastSpeech2/lexicon/librispeech-lexicon.txt")

    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = add_prefix2phone(phones, lang="Eng")
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)


def preprocess_mandarin(text, preprocess_config):
    # lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])
    lexicon = read_lexicon("/home/chungtran/Code/TTS/FastSpeech2/lexicon/pinyin-lexicon-r.txt")

    phones = []
    pinyins = [
        p[0]
        for p in pinyin(
            text, style=Style.TONE3, strict=False, neutral_tone_with_five=True
        )
    ]
    for p in pinyins:
        if p in lexicon:
            phones += lexicon[p]
        else:
            phones.append("sp")

    phones = add_prefix2phone(phones, lang="Chi")
    phones = "{" + " ".join(phones) + "}"
    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)

def preprocess_japanese(text, preprocess_config):
    fullcontext_labels = pyopenjtalk.extract_fullcontext(text)
    phones, accents = pp_symbols(fullcontext_labels)
    phones = [openjtalk2julius(p) for p in phones if p != '']
    phones = add_prefix2phone(phones, lang="Japan")

    phones = "{" + " ".join(phones) + "}"
    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)

def preprocess_vietnamese(text):
    lexicon = load_lexicon_phone_mfa("/home/chungtran/Code/TTS/montreal-forced-aligner_vn/Vietnamese_Lexicon/lexicon.txt.espeak.txt")
    words = text.split(" ")
    phones = []
    pdb.set_trace()
    for word in words:
        if word in lexicon:
            phones += lexicon[word.lower()].split()
        # else:
        #     phones.append("sp")
    phones = add_prefix2phone(phones, lang="Vie")
    phones = "{" + " ".join(phones) + "}"
    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )
    pdb.set_trace()

    return np.array(sequence)

def preprocess_audio(audio_file, _stft):
    wav, sample_rate = librosa.load(audio_file, sr=22050)
    mel_spectrogram, _ = Audio.tools.get_mel_from_wav(wav, _stft)
    return torch.from_numpy(mel_spectrogram).to(device=device)

def synthesize(model, step, configs, vocoder, batchs, control_values, args):
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
    output, postnet_output = model.inference(style_vector, src, language, src_len=src_len, max_src_len=max_src_len)
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
        default=1.0,
        help="control the speed of the whole utterance, larger value for slower speaking rate",
    )
    parser.add_argument("--ref_audio", type=str, required=True,
        help="path to an reference speech audio sample")

    args = parser.parse_args()

    # Check source texts
    # if args.mode == "batch":
    #     assert args.source is not None and args.text is None
    # if args.mode == "single":
    #     assert args.source is None and args.text is not None

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # Get model
    # model = get_model(args, configs, device, train=False)
    # model = get_model_fastSpeech2_StyleEncoder_MultiLanguage(args, configs, device, train=False)
    model = get_model_styleencoder_multispeaker(args, configs, device, train=False)
    # print(model)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("pytorch_total_params", pytorch_total_params)
    print("pytorch_total_params_trainable", pytorch_total_params_trainable)
    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "languages.json")) as f:
        language_map = json.load(f)

    # Preprocess texts
    if args.mode == "batch":
        # Get dataset
        dataset = TextDataset(args.source, preprocess_config)
        batchs = DataLoader(
            dataset,
            batch_size=8,
            collate_fn=dataset.collate_fn,
        )
    language = ["Vie"]
    # texts = [
    #     "xin chào tất cả mọi người",
    #     "大家好",
    #     "みなさん、こんにちは。",
    #     "Hi everybody",
    #     ]
    texts = [
    "truyện ngắn hai đứa trẻ xoay quanh số phận những con người nơi phố huyện nghèo qua lời kể của nhân vật liên .",
    ]
    for lang, text in zip(language, texts): 
        args.language = lang
        args.text = text
        if args.mode == "single":
            ids = raw_texts = [args.text[:100]]
            print(raw_texts)
            print("*"*20)
            print(args.text[:100])
            speakers = np.array([args.speaker_id])
            if args.language == "Vie":
                texts = np.array([preprocess_vietnamese(args.text)])
            elif args.language == "Chi":
                texts = np.array([preprocess_mandarin(args.text, preprocess_config)])
            elif args.language == "Japan":
                texts = np.array([preprocess_japanese(args.text, preprocess_config)])
            elif args.language == "Eng":
                texts = np.array([preprocess_english(args.text, preprocess_config)])
            text_lens = np.array([len(texts[0])])
            lang_id = language_map[args.language]
            batch = [ids, raw_texts, lang_id, speakers, texts, text_lens, max(text_lens)]
            print(batch)

        control_values = args.pitch_control, args.energy_control, args.duration_control
        print(args)
        synthesize(model, args.restore_step, configs, vocoder, batch, control_values, args)
