import re
import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.getcwd()), ".."))
import argparse
import os
import json
import torch
import yaml
import numpy as np

from utils.model import get_vocoder, get_model_fastSpeech2_StyleEncoder_MultiLanguage_1
from utils.tool_lexicon import processes_all_languages
from utils.model import vocoder_infer
from utils.tool_audio import preprocess_audio
import audio as Audio
import librosa
from scipy.io.wavfile import write
import random
import pdb

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

def synthesize(model, configs, vocoder, batch, control_values, args, style_vector):
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
    # write("../wav_output/{0}_{1}_corpus.wav".format(args.name, args.language), 22050, wav[0])


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
        "--corpus_path", type=str, required=True, help="path to corpus"
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

    languages = ["chinese", "dutch", "english", "french", "german",
                "indonesian", "italian", "japanese", "korean",
                "polish", "portuguese", "russian", "spanish", "vietnamese"]

    control_values = args.pitch_control, args.energy_control, args.duration_control
    corpus_path = args.corpus_path
    corpus2save = "/home/ldap-users/s2220411/Code/FastSpeech2_multilingual/corpus/Interspeech2023/Fastspeech2"

    for lang in languages:
        file_path = os.path.join(corpus_path, lang+".txt")
        with open(file_path, "r") as fread:
            selective_sentences = fread.readlines()
            # extract style speaker
            # style_vector = extract_style(model, "../audio_ref/chung_ipad.wav")

            # selective_sentences
            random.shuffle(selective_sentences)
            selective_sentences = selective_sentences[:10]

            for index, selective_sentence in enumerate(selective_sentences):
                args.language = lang
                args.text = selective_sentence.strip()
                texts = np.array([processes_all_languages(args.text, args.language, preprocess_config)])
                speakers = np.array([args.speaker_id])
                text_lens = np.array([len(texts[0])])
                lang_id = language_map[args.language]
                # batch = [ids, raw_texts, lang_id, speakers, texts, text_lens, max(text_lens)]
                batch = [lang_id, texts, text_lens, max(text_lens)]
                control_values = args.pitch_control, args.energy_control, args.duration_control
                wav = synthesize(model, configs, vocoder, batch, control_values, args, style_vector)
                name2save = f"{index}_{args.name}_{args.language}"
                write("{0}/{1}.wav".format(corpus2save, name2save), 22050, wav[0])
                with open("{0}/{1}.txt".format(corpus2save, name2save), "w") as fout:
                    fout.write(selective_sentence.strip())
                print(args.text)
                exit()