import torch
import torch.nn as nn
from torch.nn import functional as F

from .layers import LinearNorm


class Prenet(nn.Module):
    """
     Prenet
    """
    def __init__(self, in_dim, sizes):
        '''
         in_dim: n_mel_channels * n_frames_per_step
         sizes: [256, 256]
        '''
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        # in_sizes = [80*1, 256], out_sizes=sizes=[256, 256]
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=False)
             for (in_size, out_size) in zip(in_sizes, sizes)])

    def forward(self, x):
        '''
         x: (max_mel_len/1+1, batch, n_mel_channels*n_frames_per_step=80*1)
         return: (max_mel_len/1+1, batch, sizes[-1]=256)
        '''
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
        return x


