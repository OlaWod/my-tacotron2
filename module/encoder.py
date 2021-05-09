import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from .layers import ConvNorm
from mytext import symbols


class Encoder(nn.Module):
    """ Encoder """

    def __init__(self, config):
        super(Encoder, self).__init__()

        n_src_vocab = len(symbols) + 1
        
        dim = config["encoder"]["encoder_embedding_dim"]
        n_convs = config["encoder"]["encoder_n_convs"]
        kernel_size = config["encoder"]["encoder_kernel_size"]
        lsa_dim = config["decoder"]["attention_dim"]
        
        self.text_emb = nn.Embedding(n_src_vocab, dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [
                ConvNorm(
                    dim, dim, kernel_size=kernel_size, padding=(kernel_size-1)//2
                )
                for _ in range(n_convs)
            ]
        )
        self.lstm = nn.LSTM(dim, dim // 2, 1, batch_first=True, bidirectional=True)

        self.memory_layer = nn.Linear(dim, lsa_dim, bias=False)
        
    def forward(self, text, text_lens):
        '''
         text: (batch, max_text_len)
         text_lens: (batch)
         return: (batch, max_text_len, dim), (batch, max_text_len, lsa_dim)
        '''
        x = self.text_emb(text).transpose(1, 2)
        
        for conv in self.convs:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)
        x = x.transpose(1, 2)

        # pytorch tensor are not reversible, hence the conversion
        text_lens = text_lens.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, text_lens, batch_first=True)

        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)

        x, _ = nn.utils.rnn.pad_packed_sequence(
            x, batch_first=True)

        return x, self.memory_layer(x)

    def inference(self, text):
        '''
         text: (batch, max_text_len)
         return: (batch, max_text_len, dim), (batch, max_text_len, lsa_dim)
        '''
        x = self.text_emb(text).transpose(1, 2)
        
        for conv in self.convs:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)
        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)

        return x, self.memory_layer(x)
