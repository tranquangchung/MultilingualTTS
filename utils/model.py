import os
import json

import torch
import numpy as np

import hifigan
from model import FastSpeech2, FastSpeech2_StyleEncoder, FastSpeech2_StyleEncoder_Discriminator, FastSpeech2_StyleEncoder_Multilingual, FastSpeech2_StyleEncoder_Multilingual_LossStyle 
from model import FastSpeech2_MultiSpeakers_MultiLangs, FastSpeech2_StyleEncoder_HifiGan, Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator, ScheduledOptim, FastSpeech2_StyleEncoder_Multispeaker
from model import ScheduledOptim_Diffusion
from model import FastSpeech2_StyleEncoder_Multilingual_Diffusion
from model import FastSpeech2_Adaptation_Multilingualism
from model import FastSpeech2_Denoiser
from model import ECAPA_TDNN_Discriminator
from model import FastSpeech2_StyleEncoder_Multilingual_Diffusion_Style
from collections import OrderedDict
import itertools
import pathlib
import pdb

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def get_model_fastSpeech2_StyleEncoder_pretrained(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    model = FastSpeech2_StyleEncoder(preprocess_config, model_config).to(device)
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        print("load checkpoint: ", ckpt_path)
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"], strict=True)

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, args.restore_step
        )
        if args.restore_step:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model

def get_model_fastSpeech2_StyleEncoder_MultiLanguage(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    model = FastSpeech2_StyleEncoder_Multilingual(preprocess_config, model_config).to(device)
    # if args.restore_step:
    if True:
        ckpt_path=train_config["checkpoint"]["pretrained"]
        print("load checkpoint: ", ckpt_path)
        # ckpt = torch.load(ckpt_path)
        # model.load_state_dict(ckpt["model"], strict=True)

        ckpt = torch.load(ckpt_path)
        tmp = OrderedDict()
        key_reject = ["speaker_emb.weight", "encoder.src_word_emb.weight"]
        for key,val in ckpt["model"].items():
            if key not in key_reject:
                tmp[key] = val
        model.load_state_dict(tmp, strict=False)

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, args.restore_step
        )
        if args.restore_step:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model

def get_model_fastSpeech2_StyleEncoder_MultiLanguage_1(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    model = FastSpeech2_StyleEncoder_Multilingual(preprocess_config, model_config).to(device)
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, args.restore_step
        )
        if args.restore_step:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model

def get_model_fastSpeech2_StyleEncoder_MultiLanguage_Difffusion(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    model = FastSpeech2_StyleEncoder_Multilingual_Diffusion(args, preprocess_config, model_config, train_config).to(device)
    if args.restore_step:
        ckpt_path = train_config["checkpoint"]["pretrained"]
        print("load checkpoint: ", ckpt_path)
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"], strict=True)

        # print("load checkpoint: ", ckpt_path)
        # ckpt = torch.load(ckpt_path)
        # tmp = OrderedDict()
        # # key_reject = ["speaker_emb.weight", "encoder.src_word_emb.weight"]
        # key_reject = ["encoder.position_enc", "decoder.position_enc", "speaker_emb.weight", "encoder.src_word_emb.weight"]
        # for key,val in ckpt["model"].items():
        #     if key not in key_reject:
        #         tmp[key] = val
        # model.load_state_dict(tmp, strict=False)

    if train:
        scheduled_optim = ScheduledOptim_Diffusion(
            args, model, train_config, model_config, args.restore_step
        )
        if args.restore_step:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model

def get_model_fastSpeech2_StyleEncoder_MultiLanguage_Difffusion1(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    model = FastSpeech2_StyleEncoder_Multilingual_Diffusion(args, preprocess_config, model_config, train_config).to(device)
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"], strict=True)
        print("Load model: ", ckpt_path)

    if train:
        scheduled_optim = ScheduledOptim_Diffusion(
            args, model, train_config, model_config, args.restore_step
        )
        if args.restore_step:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model


def get_model_fastSpeech2_StyleEncoder_MultiLanguage_Difffusion_Style(args, configs, device, train=False):
  (preprocess_config, model_config, train_config) = configs

  model = FastSpeech2_StyleEncoder_Multilingual_Diffusion_Style(args, preprocess_config, model_config, train_config).to(
    device)
  # if args.restore_step:
  if True:
    ckpt_path = train_config["checkpoint"]["pretrained"]
    print("load checkpoint: ", ckpt_path)
    # ckpt = torch.load(ckpt_path)
    # model.load_state_dict(ckpt["model"], strict=True)

    # print("load checkpoint: ", ckpt_path)
    ckpt = torch.load(ckpt_path)
    tmp = OrderedDict()
    # key_reject = ["speaker_emb.weight", "encoder.src_word_emb.weight"]
    key_reject = ["encoder.position_enc", "decoder.position_enc", "speaker_emb.weight", "encoder.src_word_emb.weight"]
    for key,val in ckpt["model"].items():
        if key not in key_reject:
            tmp[key] = val
    model.load_state_dict(tmp, strict=False)

  if train:
    scheduled_optim = ScheduledOptim_Diffusion(
      args, model, train_config, model_config, args.restore_step
    )
    if args.restore_step:
      scheduled_optim.load_state_dict(ckpt["optimizer"])
    model.train()
    return model, scheduled_optim

  model.eval()
  model.requires_grad_ = False
  return model

def get_model_fastSpeech2_StyleEncoder_MultiLanguage_Difffusion_Style1(args, configs, device, train=False):
  (preprocess_config, model_config, train_config) = configs

  model = FastSpeech2_StyleEncoder_Multilingual_Diffusion_Style(args, preprocess_config, model_config, train_config).to(
    device)
  if args.restore_step:
      ckpt_path = os.path.join(
          train_config["path"]["ckpt_path"],
          "{}.pth.tar".format(args.restore_step),
      )
      ckpt = torch.load(ckpt_path)
      model.load_state_dict(ckpt["model"], strict=True)
      print("Load model: ", ckpt_path)

  if train:
    scheduled_optim = ScheduledOptim_Diffusion(
      args, model, train_config, model_config, args.restore_step
    )
    if args.restore_step:
      scheduled_optim.load_state_dict(ckpt["optimizer"])
    model.train()
    return model, scheduled_optim

  model.eval()
  model.requires_grad_ = False
  return model

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath)
    print("Complete.")
    return checkpoint_dict

def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def get_vocoder(configs, device):
    preprocess_config, model_config, train_config = configs
    config = model_config
    path_script = train_config["checkpoint"]["path_hifigan"]
    name = config["vocoder"]["model"]
    speaker = config["vocoder"]["speaker"]
    if name == "MelGAN":
        if speaker == "LJSpeech":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "linda_johnson"
            )
        elif speaker == "universal":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "multi_speaker"
            )
        vocoder.mel2wav.eval()
        vocoder.mel2wav.to(device)
    elif name == "HiFi-GAN":
        with open(os.path.join(path_script, "hifigan/config.json"), "r") as f:
            config = json.load(f)
        config = hifigan.AttrDict(config)
        vocoder = hifigan.Generator(config)
        if speaker == "LJSpeech":
            ckpt = torch.load(os.path.join(path_script, "hifigan/generator_LJSpeech.pth.tar"))
        elif speaker == "universal":
            ckpt = torch.load(os.path.join(path_script, "hifigan/generator_universal.pth.tar"))
        vocoder.load_state_dict(ckpt["generator"])
        vocoder.eval()
        vocoder.remove_weight_norm()
        vocoder.to(device)

    return vocoder


def vocoder_infer(mels, vocoder, model_config, preprocess_config, lengths=None):
    name = model_config["vocoder"]["model"]
    with torch.no_grad():
        if name == "MelGAN":
            wavs = vocoder.inverse(mels / np.log(10))
        elif name == "HiFi-GAN":
            wavs = vocoder(mels).squeeze(1)

    wavs = (
        wavs.cpu().numpy()
        * preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    ).astype("int16")
    wavs = [wav for wav in wavs]

    for i in range(len(mels)):
        if lengths is not None:
            wavs[i] = wavs[i][: lengths[i]]

    return wavs
