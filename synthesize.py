import re
import argparse
from string import punctuation

import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from g2p_en import G2p
from pypinyin import pinyin, Style

from utils.model import get_model, get_vocoder, get_model_pretrained, get_model_styleencoder_multispeaker, get_model_fastSpeech2_StyleEncoder_pretrained
from utils.model import get_model_adaptation_multilingualism_pretrained
from utils.tools import to_device, synth_samples
from utils.model import vocoder_infer
from dataset import TextDataset, text_to_sequence_phone_vn_mfa
from text import text_to_sequence, text_to_sequence_mfa
from text.cleaners import collapse_whitespace
import pyopenjtalk
import pdb
import audio as Audio
import librosa
from scipy.io.wavfile import write
import json
import os
import glob
import librosa
import shutil
from trash.mixing_language import get_phone_and_lang


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
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
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
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

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

    phones = "{" + " ".join(phones) + "}"
    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)

def preprocess_japanese(text:str):
    fullcontext_labels = pyopenjtalk.extract_fullcontext(text)
    phonemes, accents = pp_symbols(fullcontext_labels)
    phonemes = [openjtalk2julius(p) for p in phonemes if p != '']
    return np.array(phonemes)

def processes_vietnamese(text, preprocess_config):
    pass

def synthesize(model, step, configs, vocoder, batchs, control_values):
    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control = control_values

    for batch in batchs:
        batch = to_device(batch, device)
        with torch.no_grad():
            # Forward
            import time
            fast_speech_start = time.time()
            output = model(
                *(batch[2:]),
                p_control=pitch_control,
                e_control=energy_control,
                d_control=duration_control
            )
            fast_speech_end = time.time()
            print("Fast SPeech 2 time: ", fast_speech_end - fast_speech_start)
            synth_samples(
                batch,
                output,
                vocoder,
                model_config,
                preprocess_config,
                train_config["path"]["result_path"],
            )

def preprocess_audio(audio_file, _stft):
    wav, sample_rate = librosa.load(audio_file, sr=None)
    # if sample_rate != 16000:
    #     wav = librosa.resample(wav, sample_rate, 16000)
    if sample_rate != 22050:
        wav = librosa.resample(wav, sample_rate, 22050)
    mel_spectrogram, _ = Audio.tools.get_mel_from_wav(wav, _stft)
    return torch.from_numpy(mel_spectrogram).to(device=device)

def preprocess_audio_signal(audio, _stft):
    mel_spectrogram, _ = Audio.tools.get_mel_from_wav(audio, _stft)
    return torch.from_numpy(mel_spectrogram).to(device=device)

def synthesize_stylespeech(args, model, step, configs, vocoder, batchs, control_values):

    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control = control_values

    _stft = Audio.stft.TacotronSTFT(
            preprocess_config["preprocessing"]["stft"]["filter_length"],
            preprocess_config["preprocessing"]["stft"]["hop_length"],
            preprocess_config["preprocessing"]["stft"]["win_length"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
            preprocess_config["preprocessing"]["audio"]["sampling_rate"],
            preprocess_config["preprocessing"]["mel"]["mel_fmin"],
            preprocess_config["preprocessing"]["mel"]["mel_fmax"],
        )

    ref_mel = preprocess_audio(args.ref_audio, _stft).transpose(0,1).unsqueeze(0)
    # Extract style vector
    style_vector = model.get_style_vector(ref_mel)
    pdb.set_trace()

    # Forward
    src = torch.from_numpy(batchs[0][3]).to(device=device)
    src_len = torch.from_numpy(batchs[0][4]).to(device=device)
    speaker = torch.tensor(batchs[0][2]).to(device=device)
    max_src_len = batchs[0][5]
    # output, postnet_output = model.inference(style_vector, src, speaker, src_len=src_len, max_src_len=max_src_len)
    output, postnet_output = model.inference(style_vector, src, src_len=src_len, max_src_len=max_src_len)
    
    postnet_output = postnet_output.transpose(1, 2)
    wav = vocoder_infer(postnet_output, vocoder, model_config, preprocess_config)
    write("./wav_output/StyleSpeaker_denoiser_{0}_hifigan.wav".format(args.speaker_id), 22050, wav[0])

def get_file_style(path, speaker_name):
    files = glob.glob(os.path.join(path, speaker_name, "*.wav"))
    # files.reverse()
    for f in files:
        if "denoiser.wav" in f:
            continue
        y, sr = librosa.load(f)
        yt, index = librosa.effects.trim(y)
        duration = librosa.get_duration(yt)
        if 3.5  <  duration < 6.0:
            rf = open(f.replace(".wav", ".txt"), "r")
            text = rf.read().strip()
            text = text.lower().replace(",", " , ").replace(".", " . ").replace("?","").strip()
            text = collapse_whitespace(text)
            try:
                text_id = text_to_sequence_mfa(text)
            except Exception as e:
                # print(e)
                continue
            return yt, text
    return None, None

def synthesize_stylespeech_all_speaker(args, model, step, configs, vocoder, batchs, control_values):

    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control = control_values

    _stft = Audio.stft.TacotronSTFT(
            preprocess_config["preprocessing"]["stft"]["filter_length"],
            preprocess_config["preprocessing"]["stft"]["hop_length"],
            preprocess_config["preprocessing"]["stft"]["win_length"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
            preprocess_config["preprocessing"]["audio"]["sampling_rate"],
            preprocess_config["preprocessing"]["mel"]["mel_fmin"],
            preprocess_config["preprocessing"]["mel"]["mel_fmax"],
        )
    
    with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "speakers.json")) as f:
        speaker_map = json.load(f)
    for speaker_name, speaker_id in speaker_map.items():
        file_style, text = get_file_style(preprocess_config["path"]["raw_path"], speaker_name)
        text = """
            với hơn ba trăm ba sáu chương trình đã sản xuất và phát hành để phục vụ hàng triệu khán giả trong và ngoài nước ."""

        text = text.strip()
        print(speaker_name, text)
        if file_style is not None:
            ref_mel = preprocess_audio_signal(file_style, _stft).transpose(0,1).unsqueeze(0)
            # Extract style vector
            style_vector = model.get_style_vector(ref_mel)

            # Forward
            text_id = np.array([text_to_sequence_mfa(text)])
            text_lens = np.array([len(text_id[0])])
            src = torch.from_numpy(text_id).to(device=device)
            src_len = torch.from_numpy(text_lens).to(device=device)
            speaker = torch.tensor(np.array([speaker_id])).to(device=device)
            max_src_len = max(text_lens)
            output, postnet_output = model.inference(style_vector, src, speaker, src_len=src_len, max_src_len=max_src_len)
            
            postnet_output = postnet_output.transpose(1, 2)
            wav = vocoder_infer(postnet_output, vocoder, model_config, preprocess_config)
            write("./wav_output_denoiser/{0}_{1}_hifigan.wav".format(speaker_id, speaker_name), 22050, wav[0])
            write("./wav_output_denoiser/{0}_{1}_hifigan_gt.wav".format(speaker_id, speaker_name), 22050, file_style)


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
    parser.add_argument("--ref_audio", type=str, required=False,
        help="path to an reference speech audio sample")
    args = parser.parse_args()
    print(args)

    # Check source texts
    if args.mode == "batch":
        assert args.source is not None and args.text is None
    if args.mode == "single":
        assert args.source is None and args.text is not None

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # Get model
    # model = get_model_pretrained(args, configs, device, train=False)
    model = get_model_fastSpeech2_StyleEncoder_pretrained(args, configs, device, train=False)
    # model = get_model_adaptation_multilingualism_pretrained(args, configs, device, train=False)
    # model = get_model_styleencoder_multispeaker(args, configs, device, train=False)
    print(model)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("pytorch_total_params", pytorch_total_params)
    print("pytorch_total_params_trainable", pytorch_total_params_trainable)
    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Preprocess texts
    if args.mode == "batch":
        # Get dataset
        dataset = TextDataset(args.source, preprocess_config)
        batchs = DataLoader(
            dataset,
            batch_size=8,
            collate_fn=dataset.collate_fn,
        )
    if args.mode == "single":
        ids = raw_texts = [args.text[:100]]
        print(raw_texts)
        print("*"*20)
        print(args.text[:100])
        speakers = np.array([args.speaker_id])
        # if preprocess_config["preprocessing"]["text"]["language"] == "en":
        #     texts = np.array([preprocess_english(args.text, preprocess_config)])
        # elif preprocess_config["preprocessing"]["text"]["language"] == "zh":
        #     texts = np.array([preprocess_mandarin(args.text, preprocess_config)])
        texts = np.array([text_to_sequence_mfa(args.text)])
        text_lens = np.array([len(texts[0])])
        batchs = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens))]
        print("*"*20)
        # phone_tmp, langs_tmp = get_phone_and_lang()
        # texts = np.array([phone_tmp])
        # langs = np.array([langs_tmp])
        # text_lens = np.array([len(texts[0])])
        # batchs = [(ids, raw_texts, langs, speakers, texts, text_lens, max(text_lens))]
        # print("*"*20)
        # pdb.set_trace()
        print(batchs)

    control_values = args.pitch_control, args.energy_control, args.duration_control

    # synthesize(model, args.restore_step, configs, vocoder, batchs, control_values)
    synthesize_stylespeech(args, model, args.restore_step, configs, vocoder, batchs, control_values)
    # synthesize_stylespeech_all_speaker(args, model, args.restore_step, configs, vocoder, batchs, control_values)

