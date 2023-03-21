import sys
sys.path.append("..")
import argparse
import os
import json
import torch
import yaml
import numpy as np
from utils.model import get_vocoder, get_model_fastSpeech2_StyleEncoder_MultiLanguage_1
from utils.model import get_vocoder, get_model_fastSpeech2_StyleEncoder_MultiLanguage_1
from utils.model import get_model_fastSpeech2_StyleEncoder_MultiLanguage_Difffusion1
from utils.model import get_model_fastSpeech2_StyleEncoder_MultiLanguage_Difffusion_Style1
from utils.model import get_model_fastSpeech2_StyleEncoder_MultiLanguage_Difffusion_Style_Language1
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
from matplotlib import gridspec


plt.rcParams["figure.autolayout"] = True

import pdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
markers_dictionary = {
  "chinese": "o",
  "dutch": "8",
  "english": "v",
  "french": "<",
  "german": ">",
  "indonesian": "^",
  "italian": "D",
  "japanese": "s",
  "korean": "p",
  "polish": "P",
  "portuguese": "*",
  "russian": "h",
  "spanish": "X",
  "vietnamese": "d",
}

# markers = [".", "o", "v", "^", "1", "s", "p", "*", "+", "D"]

# markers = ["o", "s"]
# colors = ["b", "g", "r", "c", "m"]
# colors = ["b", "g", "r", "c", "m", "y", "k", "pink", "navy", "purple"]
colors = ["b", "g", "r", "m", "y", "k", "pink", "purple"]
colors_pairs = []
for color_x in colors:
    for color_y in colors:
        if color_x == color_y: continue
        colors_pairs.append([color_x, color_y])
def Calculate_PCA(data):
    model = PCA(n_components=2)
    return model.fit_transform(data)

def Calculate_TSNE(data, n_iter=5000):
    model = TSNE(n_components=2, random_state=0, n_iter=n_iter)
    return model.fit_transform(data)

def Visualization(data, name, colors_tsne=None, markers_tsne=None, colors_edge=None):
    plt.clf()
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
    ax0 = plt.subplot(gs[0])
    ax0.get_xaxis().set_visible(False)
    ax0.get_yaxis().set_visible(False)

    for i in range(data.shape[0]):
        print(data[i,:])
        x, y = data[i,:]
        ax0.plot(x, y, marker=markers_tsne[i], markersize=6, markeredgecolor=colors_edge[i],
                 markerfacecolor=colors_tsne[i], markeredgewidth=1)

    legend_elements1 = [
        Line2D([0], [0], marker='o', color='w', label='chinese', markerfacecolor='gray', markersize=7),
        Line2D([0], [0], marker='v', color='w', label='english', markerfacecolor='gray', markersize=7),
        Line2D([0], [0], marker='<', color='w', label='french', markerfacecolor='gray', markersize=7),
        Line2D([0], [0], marker='>', color='w', label='german', markerfacecolor='gray', markersize=7),
        Line2D([0], [0], marker='8', color='w', label='dutch', markerfacecolor='gray', markersize=7),
        Line2D([0], [0], marker='^', color='w', label='indonesia', markerfacecolor='gray', markersize=7),
        Line2D([0], [0], marker='D', color='w', label='italian', markerfacecolor='gray', markersize=7),
        Line2D([0], [0], marker='s', color='w', label='japanese', markerfacecolor='gray', markersize=7),
        Line2D([0], [0], marker='p', color='w', label='korean', markerfacecolor='gray', markersize=7),
        Line2D([0], [0], marker='P', color='w', label='polish', markerfacecolor='gray', markersize=7),
        Line2D([0], [0], marker='*', color='w', label='portuguese', markerfacecolor='gray', markersize=7),
        Line2D([0], [0], marker='h', color='w', label='russian', markerfacecolor='gray', markersize=7),
        Line2D([0], [0], marker='X', color='w', label='spanish', markerfacecolor='gray', markersize=7),
        Line2D([0], [0], marker='d', color='w', label='vietnamese', markerfacecolor='gray', markersize=7),
    ]
    legend1 = pyplot.legend(handles=legend_elements1, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.gca().add_artist(legend1)
    plt.tight_layout()
    # plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.savefig(f"./visualization_interspeech/{name}.png", dpi=2400)

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
    args.restore_step = 310000
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

    # #########################
    # args.restore_step = 770000
    # args.preprocess_config = "../config/config_pcgpu/LibriTTS_StyleSpeech_multilingual_diffusion_testV1/preprocess.yaml"
    # args.model_config = "../config/config_pcgpu/LibriTTS_StyleSpeech_multilingual_diffusion_testV1/model.yaml"
    # args.train_config = "../config/config_pcgpu/LibriTTS_StyleSpeech_multilingual_diffusion_testV1/train.yaml"
    # preprocess_config = yaml.load(
    #     open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    # )
    # model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    # train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    # configs = (preprocess_config, model_config, train_config)
    # model = get_model_fastSpeech2_StyleEncoder_MultiLanguage_Difffusion1(args, configs, device, train=False)
    # model_name = "fastspeech2_diffusion"
    # yield [model, model_name, configs, args]
    #
    # # # #########################
    # args.restore_step = 610000
    # args.preprocess_config = "../config/config_pcgpu/LibriTTS_StyleSpeech_multilingual_diffusion_style/preprocess.yaml"
    # args.model_config = "../config/config_pcgpu/LibriTTS_StyleSpeech_multilingual_diffusion_style/model.yaml"
    # args.train_config = "../config/config_pcgpu/LibriTTS_StyleSpeech_multilingual_diffusion_style/train.yaml"
    # preprocess_config = yaml.load(
    #     open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    # )
    # model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    # train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    # configs = (preprocess_config, model_config, train_config)
    # model = get_model_fastSpeech2_StyleEncoder_MultiLanguage_Difffusion_Style1(args, configs, device, train=False)
    # model_name = "fastspeech2_diffusion_Style"
    # yield [model, model_name, configs, args]

if __name__ == "__main__":
    path_folder = "/home/ldap-users/s2220411/Code/FastSpeech2_multilingual/visualization/synthesis_audio_interspeech2023_visualization/baseline_fastspeech2/5_second"
    # languages = os.listdir(path_folder)
    languages = ["chinese", "english", "indonesian", "japanese", "korean", "vietnamese"]
    labels = []
    embeddings = []
    colors_tsne = []
    markers_tsne = []
    colors_edge = []
    index_total_speaker = 0
    for [model, model_name, configs, args] in get_model():
        for lang in languages:
            path_folder_lang = os.path.join(path_folder, lang)
            speakers = os.listdir(path_folder_lang)
            for index_s, speaker in enumerate(speakers):
                wavs = glob.glob(os.path.join(path_folder_lang, speaker, "*.wav"))
                color_face, color_edge = colors_pairs[index_total_speaker % len(colors_pairs)]
                for wav_path in wavs:
                    o_lang, o_speaker, o_filename, n_lang, _ = wav_path.split("/")[-1].split("__")
                    print(o_lang, "-", o_speaker, "-", o_filename, "-",n_lang)
                    markers_tsne.append(markers_dictionary[n_lang.replace("_","")])
                    colors_tsne.append(color_face)
                    colors_edge.append(color_edge)
                    embedding = style_vector(model, configs, wav_path)
                    embeddings.append(embedding)
                    labels.append(speaker)
                    index_total_speaker += 1

        embeddings = np.asarray(embeddings)
        with open('./embeddings_interspeech/embeddings_langs_interspeech.npy', 'wb') as file_np:
            np.save(file_np, embeddings)
        # with open('labels.npy', 'wb') as file_np:
        #     np.save(file_np, labels)
        # with open('./embeddings_interspeech/embeddings_langs_interspeech.npy', 'rb') as file_np:
        #     embeddings = np.load(file_np)
        # with open('labels.npy', 'rb') as file_np:
        #     labels = np.load(file_np)
        print("Calculating for TSNE")
        n_iter = 5000
        print("Speaker: {0}, n_iter: {1}".format(len(labels), n_iter))
        # print(markers_tsne)
        # print(colors_tsne)
        data_tsne = Calculate_TSNE(embeddings, n_iter=n_iter)
        pdb.set_trace()
        # Visualization(data_tsne, args.name+"_TSNE", colors_tsne=colors_tsne, markers_tsne=markers_tsne, colors_edge=colors_edge)
        data_pca = Calculate_PCA(embeddings)
        # Visualization(data_pca, args.name+"_PCA", colors_tsne=colors_tsne, markers_tsne=markers_tsne, colors_edge=colors_edge)
        print("DONE")
