import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import Encoder, Decoder, PostNet
from .modules import VarianceAdaptor
from utils.tools import get_mask_from_lengths
import pdb

class FastSpeech2_Denoiser(nn.Module):
    """ FastSpeech2 """

    def __init__(self):
        super(FastSpeech2_Denoiser, self).__init__()
        self.refinement = PostNet()

    def forward(self, mels=None):
        out = self.refinement(mels)
        return out

