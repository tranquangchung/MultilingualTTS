import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import Encoder_StyleSpeech, Decoder_StyleSpeech, PostNet, Encoder_StyleSpeech_Discriminator
from .modules import VarianceAdaptor, MelStyleEncoder, VarianceAdaptor_Discriminator
from utils.tools import get_mask_from_lengths
import pdb


class FastSpeech2_StyleEncoder_Discriminator(nn.Module):
    """ FastSpeech2_StyleEncoder_Discriminator """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2_StyleEncoder_Discriminator, self).__init__()
        self.model_config = model_config
        self.melstyle_encoder = MelStyleEncoder(model_config)
        self.encoder = Encoder_StyleSpeech_Discriminator(model_config)
        self.variance_adaptor = VarianceAdaptor_Discriminator(preprocess_config, model_config)
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
        output, src_embedded, _ = self.encoder(texts, style_vector, src_masks)
        
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
            src_embedded,
            style_vector,
        )
        
    def inference(
        self,
        style_vector,
        src_seq,  # text
        src_lens, # text_lens
        max_src_len=None, # max_text_len
        return_attn=False
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        output, src_embedded, _ = self.encoder(src_seq, style_vector, src_masks)
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
            src_masks
        )

        output, mel_masks = self.decoder(output, style_vector, mel_masks)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        return (
            output,
            postnet_output,
            src_embedded,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
        )
