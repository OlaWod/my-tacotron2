import torch
import torch.nn as nn


class PostNet(nn.Module):
    """
     PostNet: Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(
        self,
        n_mel_channels=80,
        postnet_embedding_dim=512,
        postnet_kernel_size=5,
        postnet_n_blocks=5,
    ):
        super().__init__()

        self.blocks = nn.ModuleList()

        self.blocks.append(
            nn.Sequential(
                nn.Conv1d(
                    n_mel_channels,
                    postnet_embedding_dim,
                    kernel_size=postnet_kernel_size,
                    padding=(postnet_kernel_size - 1) // 2
                ),
                nn.BatchNorm1d(postnet_embedding_dim),
                nn.Tanh(),
                nn.Dropout(0.5)
            )
        )

        for i in range(1, postnet_n_blocks - 1):
            self.blocks.append(
                nn.Sequential(
                    nn.Conv1d(
                        postnet_embedding_dim,
                        postnet_embedding_dim,
                        kernel_size=postnet_kernel_size,
                        padding=int((postnet_kernel_size - 1) / 2)
                    ),
                    nn.BatchNorm1d(postnet_embedding_dim),
                    nn.Tanh(),
                    nn.Dropout(0.5)
                )
            )

        self.blocks.append(
            nn.Sequential(
                nn.Conv1d(
                    postnet_embedding_dim,
                    n_mel_channels,
                    kernel_size=postnet_kernel_size,
                    padding=int((postnet_kernel_size - 1) / 2)
                ),
                nn.BatchNorm1d(n_mel_channels),
                nn.Dropout(0.5)
            )
        )

    def forward(self, x):
        '''
         x: (batch, 80, max_mel_len)
        '''
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

        return x

