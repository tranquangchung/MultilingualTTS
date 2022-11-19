import sys
sys.path.append("..")
import argparse
import os
import json
import torch
import yaml
import numpy as np
from utils.model import get_vocoder, get_model_fastSpeech2_StyleEncoder_MultiLanguage_1
import audio as Audio
import librosa
import glob
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import pyplot
from matplotlib.patches import Patch


plt.rcParams["figure.autolayout"] = True

import pdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# markers = [".", "o", "v", "^", "1", "s", "p", "*", "+", "D"]
markers = ["o", "s"]
# colors = ["b", "g", "r", "c", "m"]
colors = ["b", "g", "r", "c", "m", "y", "k", "pink", "navy", "purple"]

def Calculate_PCA(data):
    model = PCA(n_components=2)
    return model.fit_transform(data)

def Calculate_TSNE(data, n_iter=5000):
    model = TSNE(n_components=2, random_state=0, n_iter=n_iter)
    return model.fit_transform(data)

def Visualization(data, name, colors_tsne=None, markers_tsne=None, colors_edge=None):
    plt.clf()
    for i in range(data.shape[0]):
        print(data[i,:])
        x, y = data[i,:]
        plt.plot(x, y, marker=markers_tsne[i], markersize=6, markeredgecolor=colors_edge[i], markerfacecolor=colors_tsne[i])

    legend_elements1 = [Line2D([0], [0], marker='s', color='w', label='Human', markerfacecolor='gray', markersize=7),
                       Line2D([0], [0], marker='o', color='w', label='Synthesis', markerfacecolor='gray', markersize=7)]
    legend_elements2 = [Line2D([0], [0], marker='s', color='w', label='Male', markerfacecolor='gray', markersize=7, markeredgecolor="gold"),
                       Line2D([0], [0], marker='s', color='w', label='Female', markerfacecolor='gray', markersize=7, markeredgecolor="violet")]
    # legend_elements2 = [Patch(facecolor='gray', edgecolor='gold', label='Male'),
    #                     Patch(facecolor='gray', edgecolor='violet', label='Female')]
    legend1 = pyplot.legend(handles=legend_elements1, loc=1)
    legend2 = pyplot.legend(handles=legend_elements2, loc=3)
    plt.gca().add_artist(legend1)
    plt.gca().add_artist(legend2)
    plt.savefig(f"{name}.png")

def preprocess_audio(audio_file, _stft):
    wav, sample_rate = librosa.load(audio_file, sr=22050)
    mel_spectrogram, _ = Audio.tools.get_mel_from_wav(wav, _stft)
    return torch.from_numpy(mel_spectrogram).to(device=device)

def style_vector(model, configs, audio_path):
    preprocess_config, model_config, train_config = configs
    _sftf = Audio.stft.TacotronSTFT(
        preprocess_config["preprocessing"]["stft"]["filter_length"],
        preprocess_config["preprocessing"]["stft"]["hop_length"],
        preprocess_config["preprocessing"]["stft"]["win_length"],
        preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        preprocess_config["preprocessing"]["audio"]["sampling_rate"],
        preprocess_config["preprocessing"]["mel"]["mel_fmin"],
        preprocess_config["preprocessing"]["mel"]["mel_fmax"],
    )

    wav, _ = librosa.load(audio_path, sr=22050)
    ref_mel_spectrogram, _ = Audio.tools.get_mel_from_wav(wav, _sftf)
    ref_mel = torch.from_numpy(ref_mel_spectrogram).transpose(0,1).unsqueeze(0).to(device=device)

    # Extract style vector
    style_vector = model.get_style_vector(ref_mel)
    return style_vector.cpu().detach().numpy().squeeze()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, required=True)
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="name to save",
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

    args = parser.parse_args()
    print(args)

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
    path_folder = "/home/ldap-users/s2220411/Code/FastSpeech2_multilingual/visualization/synthesis_audio"
    speaker_m = ["0000_ducnguyentrung", "0001_dungnguyentuan", "0002_nguyenthanhhai", "0005_thanhletien", "0012_hjeu"]
    speaker_f = ["0014_hongnhung", "0015_lien", "0016_minhnguyet", "0035_lethithuygiang", "0036_nguyenminhtrang"]
    labels = []
    embeddings = []
    colors_tsne = []
    markers_tsne = []
    colors_edge = []
    index_total = 0
    for index, speaker in enumerate(speaker_m):
        color_tmp = colors[index_total % len(colors)]
        wavs = glob.glob(os.path.join(path_folder, speaker, "*.wav"))
        index_total += 1
        for wav in wavs:
            print(wav)
            colors_edge.append("gold")
            colors_tsne.append(color_tmp)
            if "_synthesis.wav" in wav:
                markers_tsne.append("o")
            else:
                markers_tsne.append("s")
            labels.append(index_total)
            # embedding = style_vector(model, configs, wav)
            # embeddings.append(embedding)
    ##############################################################################
    for index, speaker in enumerate(speaker_f):
        color_tmp = colors[index_total % len(colors)]
        wavs = glob.glob(os.path.join(path_folder, speaker, "*.wav"))
        index_total += 1
        for wav in wavs:
            print(wav)
            colors_edge.append("violet")
            colors_tsne.append(color_tmp)
            if "_synthesis.wav" in wav:
                markers_tsne.append("o")
            else:
                markers_tsne.append("s")
            labels.append(index_total)
            # embedding = style_vector(model, configs, wav)
            # embeddings.append(embedding)

    # embeddings = np.asarray(embeddings)
    # with open('embeddings.npy', 'wb') as file_np:
    #     np.save(file_np, embeddings)
    # with open('labels.npy', 'wb') as file_np:
    #     np.save(file_np, labels)
    with open('embeddings.npy', 'rb') as file_np:
        embeddings = np.load(file_np)
    # with open('labels.npy', 'rb') as file_np:
    #     labels = np.load(file_np)
    print("Calculating for TSNE")
    n_iter = 5000
    print("Speaker: {0}, n_iter: {1}".format(len(labels), n_iter))
    print(markers_tsne)
    print(colors_tsne)
    data_tsne = Calculate_TSNE(embeddings, n_iter=n_iter)
    Visualization(data_tsne, args.name+"_TSNE", colors_tsne=colors_tsne, markers_tsne=markers_tsne, colors_edge=colors_edge)
    data_pca = Calculate_PCA(embeddings)
    Visualization(data_pca, args.name+"_PCA", colors_tsne=colors_tsne, markers_tsne=markers_tsne, colors_edge=colors_edge)
    print("DONE")
