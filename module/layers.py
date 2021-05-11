import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


class ConvNorm(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        w_init_gain='linear'
    ):
        super().__init__()

        self.conv1d = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

        torch.nn.init.xavier_uniform_(
            self.conv1d.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        '''
         x: (batch, length, in_channels)
        '''
        x = self.conv1d(x)

        return x


class LinearNorm(nn.Module):
    
    def __init__(
        self,
        in_dim,
        out_dim,
        bias=True,
        w_init_gain='linear'
    ):
        super().__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)
