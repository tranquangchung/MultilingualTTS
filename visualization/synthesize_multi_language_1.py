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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

abbv_language = {
    "dutch": "nl",
    "french": "fr-fr",
    "german": "de",
    "indonesian": "id",
    "italian": "it",
    "korean": "ko",
    "polish": "pl",
    "portuguese": "pt",
    "russian": "ru",
    "spanish": "es",
}

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

def load_lexicon_phone_mfa(filename):
    lexicon_phone = dict()
    with open(filename, encoding='utf-8') as f:
        for index, line in enumerate(f):
            lexicon, phone = line.strip().split("\t")
            lexicon_phone[lexicon] = phone.strip()
    return lexicon_phone

def preprocess_english(text, preprocess_config):
    text = text.rstrip(punctuation)
    lexicon = read_lexicon("lexicon/dictionary/english-lexicon.txt")

    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = add_prefix2phone(phones, lang="english")
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")

    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)


def preprocess_mandarin(text, preprocess_config):
    lexicon = read_lexicon("lexicon/dictionary/pinyin-lexicon-r.txt")

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

    phones = add_prefix2phone(phones, lang="chinese")
    phones = "{" + " ".join(phones) + "}"
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
    phones = add_prefix2phone(phones, lang="japanese")

    phones = "{" + " ".join(phones) + "}"
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)

def preprocess_language(text, preprocess_config, language):
    lexicon = load_lexicon_phone_mfa(f"lexicon/dictionary/{language}_espeak.txt")
    words = text.split(" ")
    phones = []
    for word in words:
        if word in lexicon:
            phones += lexicon[word.lower()].split()
        else:
            command = "echo {0} | phonemize -l {1} -b espeak --language-switch remove-flags -p '-' -s '|' --strip".format(word, abbv_language[language])
            phoneme = subprocess.getstatusoutput(command)[1].split("\n")[-1]
            phoneme = phoneme.replace("-", " ")
            phoneme = phoneme.strip().split()
            phones += phoneme
            print("Out of Vocabulary: ",word, phoneme)
    phones = add_prefix2phone(phones, lang=language)
    phones = "{" + " ".join(phones) + "}"
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)

def preprocess_audio(audio_file, _stft):
    wav, sample_rate = librosa.load(audio_file, sr=22050)
    mel_spectrogram, _ = Audio.tools.get_mel_from_wav(wav, _stft)
    return torch.from_numpy(mel_spectrogram).to(device=device)

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

    with open("./duration/lt_5second.txt", "w") as fout:
        for lang in language:
            path_lang = os.path.join(raw_path, lang)
            speakers = os.listdir(path_lang)
            args.language = lang
            total_speaker_index = 0
            for speaker in speakers:
                path_speaker = os.path.join(path_lang, speaker)
                wavs = glob.glob(os.path.join(path_speaker, "*.wav"))
                total_wav_file = []
                for wav_path in wavs:
                    file_name = wav_path.split("/")[-1].split(".")[0]
                    wav, _ = librosa.load(wav_path)
                    wav = librosa.effects.trim(wav)[0]
                    duration = librosa.get_duration(wav)
                    ############ processing ###########
                    if 5 < duration:
                        item = [lang, speaker, file_name, "%.2f"%duration]
                        print(item)
                        total_wav_file.append(item)
                    if len(total_wav_file) >= 5:
                        break
                if len(total_wav_file) >= 5:
                    for line in total_wav_file[:5]:
                        i_lang, i_speaker, i_filename, i_duration = lang, speaker, file_name, duration
                        text_line = "{0}|{1}|{2}|{3}\n".format(i_lang, i_speaker, i_filename, i_duration)
                        fout.write(text_line)
                    total_speaker_index += 1
                if total_speaker_index == 5: break

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
