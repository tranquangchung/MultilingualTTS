import torch
import numpy as np
import os
import argparse
import librosa
import re
import json
from string import punctuation
from g2p_en import G2p

# from models.StyleSpeech import StyleSpeech
from utils.model import get_model_fastSpeech2_StyleEncoder_pretrained
from text import text_to_sequence
import audio as Audio
import utils
import pdb
from scipy.io.wavfile import write
from vocoder.inference import infer_waveform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def preprocess_english(text, lexicon_path):
    text = text.rstrip(punctuation)
    lexicon = read_lexicon(lexicon_path)

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
    sequence = np.array(text_to_sequence(phones, ['english_cleaners']))

    return torch.from_numpy(sequence).to(device=device)


def preprocess_audio(audio_file, _stft):
    wav, sample_rate = librosa.load(audio_file, sr=None)
    print(sample_rate)
    # if sample_rate != 16000:
    #     wav = librosa.resample(wav, sample_rate, 16000)
    if sample_rate != 22050:
        wav = librosa.resample(wav, sample_rate, 22050)
    mel_spectrogram, _ = Audio.tools.get_mel_from_wav(wav, _stft)
    return torch.from_numpy(mel_spectrogram).to(device=device)


def get_StyleSpeech(config, checkpoint_path):
    model = StyleSpeech(config).to(device=device)
    model.load_state_dict(torch.load(checkpoint_path)['model'])
    model.eval()
    return model


def synthesize(args, model, _stft): 
    # preprocess audio and text
    ref_mel = preprocess_audio(args.ref_audio, _stft).transpose(0,1).unsqueeze(0)
    src = preprocess_english(args.text, args.lexicon_path).unsqueeze(0)
    src_len = torch.from_numpy(np.array([src.shape[1]])).to(device=device)
    
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    # Extract style vector
    style_vector = model.get_style_vector(ref_mel)

    # Forward
    mel_output = model.inference(style_vector, src, src_len)[0]
    
    mel_ref_ = ref_mel.cpu().squeeze().transpose(0, 1).detach()
    mel_ = mel_output.cpu().squeeze().transpose(0, 1).detach()
    # plotting
    utils.plot_data([mel_ref_.numpy(), mel_.numpy()], 
        ['Ref Spectrogram', 'Synthesized Spectrogram'], filename=os.path.join(save_path, 'plot.png'))
    print('Generate done!')
    print("*"*20)
    # vocoder = torch.jit.load('/home/chungtran/Code/TTS/hifi-gan-jit/output_jit/hifigan_cpu.pt').to("cpu")
    # vocoder.eval()
    # mel_ =  mel_.unsqueeze(0).to("cpu")
    # wav = vocoder(mel_)
    # wav = wav.squeeze()
    # MAX_WAV_VALUE = 32768.0
    # wav = wav * MAX_WAV_VALUE
    # wav = wav.detach().cpu().numpy().astype('int16')
    # # write(f"./mel_0_hifigan.wav", 22050, wav)
    # write("./audio_demo/{0}_synthesis_22050.wav".format(args.name), 22050, wav)
    ###################################
    # vocoder = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')
    # # vocoder.inverse(audio) 
    # tmp = mel_output.squeeze().detach().cpu().numpy()

    # mel_ =  mel_.unsqueeze(0).to("cpu")
    # recons = vocoder.inverse(mel_).squeeze().cpu().numpy()
    # MAX_WAV_VALUE = 32768.0
    # recons = (recons*MAX_WAV_VALUE).astype('int16')
    # write("./audio_demo/{0}_synthesis_new.wav".format(args.name), 16000, recons)
    ###################################
    # mel_ =  mel_.unsqueeze(0).to("cpu")
    # recons = infer_waveform(mel_)
    # MAX_WAV_VALUE = 32768.0
    # recons = (recons*MAX_WAV_VALUE).astype('int16')
    # write("./audio_demo/p237_synthesis_wavernn.wav", 16000, recons)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True, 
        help="Path to the pretrained model")
    parser.add_argument('--config', default='configs/config.json')
    parser.add_argument("--save_path", type=str, default='results/')
    parser.add_argument("--ref_audio", type=str, required=True,
        help="path to an reference speech audio sample")
    parser.add_argument("--name", type=str, required=True,
        help="name to save audio")
    parser.add_argument("--text", type=str, required=True,
        help="raw text to synthesize")
    parser.add_argument("--lexicon_path", type=str, default='lexicon/librispeech-lexicon.txt')
    args = parser.parse_args()
    
    with open(args.config) as f:
        data = f.read()
    json_config = json.loads(data)
    config = utils.AttrDict(json_config)

    # Get model
    model = get_model_fastSpeech2_StyleEncoder_pretrained(args, configs, device, train=False)
    print(model)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("pytorch_total_params", pytorch_total_params)
    print("pytorch_total_params_trainable", pytorch_total_params_trainable)
    print('model is prepared')

    _stft = Audio.stft.TacotronSTFT(
                config.filter_length,
                config.hop_length,
                config.win_length,
                config.n_mel_channels,
                config.sampling_rate,
                config.mel_fmin,
                config.mel_fmax)

    # Synthesize
    synthesize(args, model, _stft)
