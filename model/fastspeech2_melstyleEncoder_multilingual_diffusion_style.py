import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import Encoder_StyleSpeech, Decoder_StyleSpeech, PostNet
from .modules import VarianceAdaptor, MelStyleEncoder
from utils.tools import get_mask_from_lengths
from .diffusion import GaussianDiffusion, GaussianDiffusionShallow, GaussianDiffusionShallowStyle
from .modules_diffusion import FastspeechEncoder, FastspeechDecoder
import pdb


class FastSpeech2_StyleEncoder_Multilingual_Diffusion_Style(nn.Module):
  """ FastSpeech2_StyleEncoder_Multilingual Diffusion"""

  def __init__(self, args, preprocess_config, model_config, train_config):
    super(FastSpeech2_StyleEncoder_Multilingual_Diffusion_Style, self).__init__()
    self.model = args.model
    self.model_config = model_config
    self.melstyle_encoder = MelStyleEncoder(model_config)
    self.encoder = Encoder_StyleSpeech(model_config)
    self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)

    ###############
    self.diffusion = None
    if self.model in ["aux", "shallow", "shallowstyle"]:
      # self.decoder = FastspeechDecoder(model_config)
      self.decoder = Decoder_StyleSpeech(model_config)
      self.mel_linear = nn.Linear(
        model_config["transformer"]["decoder_hidden"],
        preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
      )
      self.diffusion = GaussianDiffusionShallowStyle(preprocess_config, model_config, train_config)
    else:
      raise NotImplementedError
    ###############
    # self.decoder = Decoder_StyleSpeech(model_config)
    # self.mel_linear = nn.Linear(
    #   model_config["transformer"]["decoder_hidden"],
    #   preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
    # )
    # self.postnet = PostNet()

    self.speaker_emb = None
    if model_config["multi_language"]:
      with open(
          os.path.join(
            preprocess_config["path"]["preprocessed_path"], "languages.json"
          ),
          "r",
      ) as f:
        n_language = len(json.load(f))
      self.language_emb = nn.Embedding(
        n_language,
        model_config["transformer"]["encoder_hidden"],
      )

  def forward(
      self,
      languages,
      speakers,
      texts,
      src_lens,
      max_src_len,
      mels=None,
      mel_lens=None,
      max_mel_len=None,
      p_targets=None,
      e_targets=None,
      d_targets=None,
      p_control=1.0,
      e_control=1.0,
      d_control=1.0,
  ):
    src_masks = get_mask_from_lengths(src_lens, max_src_len)
    mel_masks = (
      get_mask_from_lengths(mel_lens, max_mel_len)
      if mel_lens is not None
      else None
    )
    style_vector = self.melstyle_encoder(mels, mel_masks)
    output = self.encoder(texts, style_vector, src_masks)
    if self.language_emb is not None:
      output = output + self.language_emb(languages).unsqueeze(1).expand(
        -1, max_src_len, -1
      )
    (
      output,
      p_predictions,
      e_predictions,
      log_d_predictions,
      d_rounded,
      mel_lens,
      mel_masks,
    ) = self.variance_adaptor(
      output,
      src_masks,
      mel_masks,
      max_mel_len,
      p_targets,
      e_targets,
      d_targets,
      p_control,
      e_control,
      d_control,
    )
    output_diffusion=None
    loss = None
    if self.model in ["aux", "shallow", "shallowstyle"]:
      epsilon_predictions = noise_loss = diffusion_step = None
      cond = output.clone()
      # output, _ = self.decoder(output, style_vector, mel_masks)
      output, mel_masks = self.decoder(output, style_vector, mel_masks)
      # output = self.decoder(output, mel_masks)
      output = self.mel_linear(output)
      self.diffusion.aux_mel = output.clone()

      if self.model in ["shallow", "shallowstyle"]:
        (output_diffusion, epsilon_predictions, loss, diffusion_step,) = self.diffusion(mels, cond, style_vector, mel_masks,)
    else:
      raise NotImplementedError
    return (
      output, # 0
      output_diffusion, # 1
      epsilon_predictions, # 2
      loss, # 3
      diffusion_step, # 3
      p_predictions, # 4
      e_predictions, # 5
      log_d_predictions, # 6
      d_rounded, # 7
      src_masks, # 8
      mel_masks, # 9
      src_lens, # 10
      mel_lens, # 11
    )
  #
  # def get_style_vector(self, mel_target, mel_len=None):
  #   mel_mask = get_mask_from_lengths(mel_len) if mel_len is not None else None
  #   style_vector = self.melstyle_encoder(mel_target, mel_mask)
  #   return style_vector
  #
  # def inference(self, style_vector, src_seq, language, src_len=None, max_src_len=None, return_attn=False,
  #               p_control=1.0, e_control=1.0, d_control=1.0, K_step=30):
  #   src_mask = get_mask_from_lengths(src_len, max_src_len)
  #   # Encoder
  #   output = self.encoder(src_seq, style_vector, src_mask)
  #
  #   if self.language_emb is not None:
  #     output = output + self.language_emb(language).unsqueeze(1).expand(
  #       -1, max_src_len, -1
  #     )
  #
  #   (
  #     output,
  #     p_predictions,
  #     e_predictions,
  #     log_d_predictions,
  #     d_rounded,
  #     mel_lens,
  #     mel_masks,
  #   ) = self.variance_adaptor(
  #     output,
  #     src_mask,
  #     p_control=p_control,
  #     e_control=e_control,
  #     d_control=d_control,
  #   )
  #   mels = None # only for inference
  #   if self.model == "naive":
  #     (output, epsilon_predictions, loss, diffusion_step,) = self.diffusion(mels, output, mel_masks,)
  #   elif self.model in ["aux", "shallow"]:
  #     epsilon_predictions = noise_loss = diffusion_step = None
  #     cond = output.clone()
  #     output, _ = self.decoder(output, style_vector, mel_masks)
  #     output = self.mel_linear(output)
  #     self.diffusion.aux_mel = output.clone()
  #     if self.model == "shallow":
  #       (output_diffusion, epsilon_predictions, loss, diffusion_step,) = self.diffusion.forward1(mels, cond, mel_masks, K_step)
  #   else:
  #     raise NotImplementedError
  #
  #   return (
  #     output_diffusion, # 0
  #     output_diffusion# epsilon_predictions, # 1
  #   )
