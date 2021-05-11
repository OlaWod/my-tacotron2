import torch
import torch.nn as nn
from torch.nn import functional as F

from .layers import ConvNorm, LinearNorm


class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size, attention_dim):
        super(LocationLayer, self).__init__()
        self.location_conv = ConvNorm(2, attention_n_filters,
                                      kernel_size=attention_kernel_size,
                                      padding=int((attention_kernel_size - 1) / 2),
                                      bias=False)
        self.location_dense = LinearNorm(attention_n_filters, attention_dim,
                                         bias=False, w_init_gain='tanh')

    def forward(self, attention_weights_cat):
        processed = self.location_conv(attention_weights_cat)
        processed = processed.transpose(1, 2)
        processed = self.location_dense(processed)
        return processed


class LSA(nn.Module):
    """
     Location-Sensitive Attention
    """
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        super(LSA, self).__init__()
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
        self.w = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)

    def get_alignment(self, query, processed_memory,
                               attention_weights_cat):
        """
        PARAMS
        ------
        query: decoder output (batch, n_mel_channels * n_frames_per_step)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)
        RETURNS
        -------
        alignment (batch, max_time)
        """

        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        alignment = self.w(torch.tanh(
            processed_query + processed_attention_weights + processed_memory))

        alignment = alignment.squeeze(-1)
        return alignment

    def forward(self, attention_rnn_h, memory, processed_memory,
                attention_weights_cat, memory_mask):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """
        alignment = self.get_alignment(
            attention_rnn_h, processed_memory, attention_weights_cat)

        if memory_mask is not None:
            alignment.masked_fill_(memory_mask, -float("inf"))

        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights
    
