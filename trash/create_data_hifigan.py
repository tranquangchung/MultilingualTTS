import argparse
import os

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.model import get_model, get_vocoder, get_param_num, get_model_fastSpeech2_StyleEncoder, get_model_fastSpeech2_StyleEncoder_MultiLanguage
from utils.tools import to_device, log, synth_one_sample, synth_one_sample_multilingual
from model import FastSpeech2Loss_MultiLingual
from dataset_multi import Dataset
from scipy.io.wavfile import write
from utils.model import vocoder_infer
import pdb
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args, configs):
    print("Prepare training ...")

    preprocess_config, model_config, train_config = configs

    # Get dataset
    dataset = Dataset(
        "train.txt", preprocess_config, train_config, sort=False, drop_last=False
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    # Prepare model
    model = get_model_fastSpeech2_StyleEncoder_MultiLanguage(args, configs, device, train=False)
    num_param = get_param_num(model)
    Loss = FastSpeech2Loss_MultiLingual(preprocess_config, model_config).to(device)
    print("Number of FastSpeech2 Parameters:", num_param)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)
    path_save = "/data/processed/speech/tts/Hifigan"
    # Init logger
    post_fix = "Full_Lang_add_prefix_500000"
    for batchs in loader:
        for batch in batchs:
            name = batch[0][0]
            speaker = name.split("_")[0]
            batch = to_device(batch, device)
            # Extract style vector
            mel_lens = batch[8].item()  
            ref_mel = batch[7]
            style_vector = model.get_style_vector(ref_mel)
            # Forward
            src = batch[4]
            src_len = batch[5]
            language = batch[2]
            max_src_len = batch[6]
            output, postnet_output = model.inference(style_vector, src, language, src_len=src_len, max_src_len=max_src_len)
            post_mel_lens = postnet_output.shape[1]  
            if mel_lens == post_mel_lens:
                pass
            elif mel_lens > post_mel_lens:
                ref_mel = ref_mel[0,:post_mel_lens,:]
            elif post_mel_lens > mel_lens:
                postnet_output = postnet_output[0,:mel_lens,:]
            ref_mel_len = ref_mel.shape[0]
            if ref_mel_len < 60:
                continue
            ref_mel = ref_mel.cpu().detach().numpy().squeeze()
            postnet_output = postnet_output.cpu().detach().numpy().squeeze()
            save_file_synthesis = "{0}--{1}.npy".format(name, post_fix)
            save_file = "{0}.npy".format(name)
            np.save(os.path.join(path_save , "mel_synthesis", save_file_synthesis), postnet_output)
            np.save(os.path.join(path_save, "mel_gt", save_file), ref_mel)

            # postnet_output = postnet_output.transpose(1, 2)
            # wav = vocoder_infer(postnet_output, vocoder, model_config, preprocess_config)
            # write("./wav_output_tmp/{0}-mel-{1}.wav".format(speaker, name), 22050, wav[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
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

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    main(args, configs)
