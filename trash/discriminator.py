import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
import pdb

class StyleDiscriminator(nn.Module):
    ''' Style Discriminator '''
    def __init__(self):
        super(StyleDiscriminator, self).__init__()
        self.conv1d = nn.Sequential(
                Conv1d(80, 512, 7, 1),
                nn.LeakyReLU(0.2, True),
                Conv1d(512, 512, 7, 1),
                nn.LeakyReLU(0.2, True),
                Conv1d(512, 512, 7, 1),
                nn.LeakyReLU(0.2, True)
                )

    def forward(self, mels, mels_gt):
        # Style discriminator
        mels = mels.permute(0, 2, 1)
        x = self.conv1d(mels)
        print(x.shape)
        print(x.max(), x.min())
        

        return None

if __name__ == "__main__":
    discirminator = StyleDiscriminator()
    mels = torch.randn(2, 100, 80)
    mels_gt = torch.randn(2, 100, 80)
    tmp = discirminator(mels, mels_gt)
    pass
