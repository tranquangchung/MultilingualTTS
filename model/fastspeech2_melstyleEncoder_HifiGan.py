import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import Encoder_StyleSpeech, Decoder_StyleSpeech, PostNet
from .modules import VarianceAdaptor, MelStyleEncoder
from utils.tools import get_mask_from_lengths
import pdb


class FastSpeech2_StyleEncoder_HifiGan(nn.Module):
    """ FastSpeech2_StyleEncoder """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2_StyleEncoder_HifiGan, self).__init__()
        self.model_config = model_config
        # self.melstyle_encoder = MelStyleEncoder(model_config)
        self.encoder = Encoder_StyleSpeech(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
        self.decoder = Decoder_StyleSpeech(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.melstyle_encoder = nn.Linear(
            8192,
            model_config["mel_style"]["style_vector_dim"],
        )
        self.postnet = PostNet()

        self.speaker_emb = None
        # if model_config["multi_speaker"]:
        #     with open(
        #         os.path.join(
        #             preprocess_config["path"]["preprocessed_path"], "speakers.json"
        #         ),
        #         "r",
        #     ) as f:
        #         n_speaker = len(json.load(f))
        #     self.speaker_emb = nn.Embedding(
        #         n_speaker,
        #         model_config["transformer"]["encoder_hidden"],
        #     )
        
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
        languages, #0
        speakers, #1
        texts, #2
        src_lens, #3
        max_src_len, #4
        mels=None, #5
        mel_lens=None , #6
        max_mel_len=None, #7
        p_targets=None, #8
        e_targets=None, #9
        d_targets=None, #10
        p_control=1.0, #11
        e_control=1.0, #12
        d_control=1.0, #13
        style_vector=None #14
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )
        style_vector_tmp = self.melstyle_encoder(style_vector)
        # style_vector_tmp = self.melstyle_encoder(mels, mel_masks)
        output = self.encoder(texts, style_vector_tmp, src_masks)

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

        output, mel_masks = self.decoder(output, style_vector_tmp, mel_masks)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
        )
       
    # def get_style_vector(self, mel_target, mel_len=None):
    #     mel_mask = get_mask_from_lengths(mel_len) if mel_len is not None else None
    #     style_vector = self.melstyle_encoder(mel_target, mel_mask)
    #     return style_vector

    # def inference(self, style_vector, src_seq, language, src_len=None, max_src_len=None, return_attn=False):
    #     src_mask = get_mask_from_lengths(src_len, max_src_len)
    #     # Encoder
    #     output = self.encoder(src_seq, style_vector, src_mask)

    #     if self.language_emb is not None:
    #         output = output + self.language_emb(language).unsqueeze(1).expand(
    #             -1, max_src_len, -1
    #         )

    #     (
    #         output,
    #         p_predictions,
    #         e_predictions,
    #         log_d_predictions,
    #         d_rounded,
    #         mel_lens,
    #         mel_masks,
    #     ) = self.variance_adaptor(
    #         output,
    #         src_mask,
    #     )

    #     output, mel_masks = self.decoder(output, style_vector, mel_masks)
    #     output = self.mel_linear(output)

    #     postnet_output = self.postnet(output) + output

    #     return (
    #         output,
    #         postnet_output,
    #     )
