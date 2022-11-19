import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import Encoder_StyleSpeech, Decoder_StyleSpeech, PostNet
from .modules import VarianceAdaptor, MelStyleEncoder
from utils.tools import get_mask_from_lengths
import pdb


class FastSpeech2_StyleEncoder(nn.Module):
    """ FastSpeech2_StyleEncoder """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2_StyleEncoder, self).__init__()
        self.model_config = model_config
        self.melstyle_encoder = MelStyleEncoder(model_config)
        self.encoder = Encoder_StyleSpeech(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
        self.decoder = Decoder_StyleSpeech(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.postnet = PostNet()

        self.speaker_emb = None
        if model_config["multi_speaker"]:
            with open(
                os.path.join(
                    preprocess_config["path"]["preprocessed_path"], "speakers.json"
                ),
                "r",
            ) as f:
                n_speaker = len(json.load(f))
            self.speaker_emb = nn.Embedding(
                n_speaker,
                model_config["transformer"]["encoder_hidden"],
            )

    def forward(
        self,
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

        # if self.speaker_emb is not None:
        #     output = output + self.speaker_emb(speakers).unsqueeze(1).expand(
        #         -1, max_src_len, -1
        #     )

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

        output, mel_masks = self.decoder(output, style_vector, mel_masks)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        return (
            output, # 0
            postnet_output, # 1
            p_predictions, # 2
            e_predictions, # 3
            log_d_predictions, # 4
            d_rounded, # 5
            src_masks, # 6
            mel_masks, # 7
            src_lens, # 8
            mel_lens, # 9
        )
       
    def get_style_vector(self, mel_target, mel_len=None):
        mel_mask = get_mask_from_lengths(mel_len) if mel_len is not None else None
        style_vector = self.melstyle_encoder(mel_target, mel_mask)
        return style_vector

    def inference(self, style_vector, src_seq, src_len=None, max_src_len=None, return_attn=False):
        src_mask = get_mask_from_lengths(src_len, max_src_len)
        
        # Encoder
        output = self.encoder(src_seq, style_vector, src_mask)
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
            src_mask,
        )

        output, mel_masks = self.decoder(output, style_vector, mel_masks)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        return (
            output,
            postnet_output,
        )
