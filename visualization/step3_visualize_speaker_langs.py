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
import torchaudio
from speechbrain.pretrained import EncoderClassifier

plt.rcParams["figure.autolayout"] = True

import pdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
markers_dictionary = {
  "chinese": "o",
  "dutch": "8",
  "english": "v",
  "french": "<",
  "german": ">",
  "indonesian": "X",
  "italian": "D",
  "japanese": "s",
  "groundtruth": "p",
  "polish": "P",
  "portuguese": "*",
  "russian": "h",
  "spanish": "X",
  "vietnamese": "d",
}
# colors = ["b", "g", "r", "m", "y", "k", "pink", "purple"]
# colors_pairs = []
# for color_x in colors:
#     for color_y in colors:
#         if color_x == color_y: continue
#         colors_pairs.append([color_x, color_y])
colors_pairs = []
# cm = plt.cm.get_cmap('tab20')
colors = ["b", "g", "r", "y", "m"]
for idx, rgb in enumerate(colors):
    colors_pairs.append([rgb, [0,0,0]])

def Calculate_PCA(data):
    model = PCA(n_components=2)
    return model.fit_transform(data)

def Calculate_TSNE(data, n_iter=5000):
    model = TSNE(n_components=2, random_state=0, n_iter=n_iter)
    return model.fit_transform(data)

def Visualization(data, labels, name, colors_tsne=None, markers_tsne=None, colors_edge=None):
    plt.clf()
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
    ax0 = plt.subplot(gs[0])
    # ax0.get_xaxis().set_visible(False)
    # ax0.get_yaxis().set_visible(False)

    for i in range(data.shape[0]):
        # print(data[i,:])
        x, y = data[i,:]
        ax0.plot(x, y, marker=markers_tsne[i], markersize=15, markeredgecolor=colors_edge[i],
                 markerfacecolor=colors_tsne[i], markeredgewidth=1)

    legend_elements1 = [
        Line2D([0], [0], marker='o', color='w', label='Chinese', markerfacecolor='black', markersize=7),
        Line2D([0], [0], marker='v', color='w', label='English', markerfacecolor='black', markersize=7),
        Line2D([0], [0], marker='X', color='w', label='Indonesia', markerfacecolor='black', markersize=7),
        Line2D([0], [0], marker='s', color='w', label='Japanese', markerfacecolor='black', markersize=7),
        Line2D([0], [0], marker='d', color='w', label='Vietnamese', markerfacecolor='black', markersize=7),
        Line2D([0], [0], marker='p', color='w', label='Ground truth', markerfacecolor='black', markersize=7),
    ]
    legend1 = pyplot.legend(handles=legend_elements1, bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0., edgecolor="black")
    plt.gca().add_artist(legend1)

    legend_elements2 = []
    for speaker, color in labels:
        tmp = Line2D([0], [0], marker='o', color='w', label=speaker, markerfacecolor=color, markersize=8)
        legend_elements2.append(tmp)
    legend2 = pyplot.legend(handles=legend_elements2, bbox_to_anchor=(1.01, 0.5), loc=2, borderaxespad=0., edgecolor="black")
    plt.gca().add_artist(legend2)

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
    args.restore_step = 100000
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
    # #########################
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
    path_folder = "/home/ldap-users/s2220411/Code/FastSpeech2_multilingual/visualization/audio/synthesis_audio_interspeech2023_visualization3_5languagesV2/"
    # languages = os.listdir(path_folder)
    # languages = ["chinese", "english", "indonesian", "korean", "japanese", "vietnamese"]
    languages = ["chinese", "english", "indonesian", "japanese", "vietnamese"]
    languages_shortcut = {
        "chinese": "ZH",
        "english": "EN",
        "indonesian": "ID",
        "japanese": "JP",
        "vietnamese": "VN"
    }
    iteration_model = {
        "baseline_fastspeech2": 1000,
        "fastspeech2_diffusion": 1000,
        "fastspeech2_diffusion_Style": 1000,
    }
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb",
                                               run_opts={"device":"cuda"})
    speakers_visualization = ["SSB0122", "1028", "Ind001", "jvs004", "0012_hjeu"]
    for [model, model_name, configs, args] in get_model():
        labels = []
        embeddings = []
        colors_tsne = []
        markers_tsne = []
        colors_edge = []
        index_total_speaker = 0
        for lang in languages:
            path_folder_lang = os.path.join(path_folder, model_name, "3_second", lang)
            speakers = os.listdir(path_folder_lang)
            for index_s, speaker in enumerate(speakers):
                if speaker not in speakers_visualization: continue
                wavs = glob.glob(os.path.join(path_folder_lang, speaker, "*.wav"))
                color_face, color_edge = colors_pairs[index_total_speaker % len(colors_pairs)]
                for wav_path in wavs:
                    synthesized_lang, speaker, filename, lang, type_, _ = wav_path.split("/")[-1].split("__")
                    if synthesized_lang == "korean": continue
                    if type_ == "groundtruth":
                        markers_tsne.append(markers_dictionary[type_.replace("_","")])
                    if type_ == "synthesis":
                        markers_tsne.append(markers_dictionary[synthesized_lang.replace("_", "")])
                    print(synthesized_lang, "-", speaker, "-", filename, "-",lang, '-', type_)
                    colors_tsne.append(color_face)
                    colors_edge.append(color_edge)
                    #############
                    signal, fs = torchaudio.load(wav_path)
                    embedding = classifier.encode_batch(signal)
                    embedding = embedding.cpu().detach().numpy().squeeze()
                    embeddings.append(embedding)
                    speaker_label = f"{speaker}({languages_shortcut[lang]})"

                    if [speaker_label, color_face] not in labels:
                        labels.append([speaker_label, color_face])
                index_total_speaker += 1
                break

        embeddings = np.asarray(embeddings)
        # with open(f'./embeddings_interspeech/{model_name}_interspeech2023.npy', 'wb') as file_np:
        #     np.save(file_np, embeddings)
        # with open(f'./embeddings_interspeech/{model_name}_label_interspeech2023.npy', 'wb') as file_np:
        #     np.save(file_np, labels)
        # with open(f'./embeddings_interspeech/{model_name}_interspeech2023.npy', 'rb') as file_np:
        #     embeddings = np.load(file_np)
        # with open(f'./embeddings_interspeech/{model_name}_label_interspeech2023.npy', 'rb') as file_np:
        #     labels = np.load(file_np)
        print("Calculating for TSNE")
        n_iter = iteration_model[model_name]
        print("Speaker: {0}, n_iter: {1}".format(len(labels), n_iter))
        # print(markers_tsne)
        # print(colors_tsne)
        data_tsne = Calculate_TSNE(embeddings, n_iter=n_iter)
        min_value = data_tsne.min()
        max_value = data_tsne.max()
        Visualization(data_tsne, labels, model_name+"_TSNE", colors_tsne=colors_tsne, markers_tsne=markers_tsne, colors_edge=colors_edge)
        # data_pca = Calculate_PCA(embeddings)
        # Visualization(data_pca, model_name+"_PCA", colors_tsne=colors_tsne, markers_tsne=markers_tsne, colors_edge=colors_edge)
        print("DONE")