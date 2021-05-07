import os
import torch
import numpy as np
from torch.nn import functional as F


def to_device(data, device):        
    (
        text_padded,
        input_lengths,
        mel_padded,
        gate_padded,
        output_lengths
    ) = data

    text_padded = text_padded.to(device)
    input_lengths = input_lengths.to(device)
    mel_padded = mel_padded.to(device)
    gate_padded = gate_padded.to(device)
    output_lengths = output_lengths.to(device)

    return (
        text_padded,
        input_lengths,
        mel_padded,
        gate_padded,
        output_lengths
    )


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len).to(lengths.device)
    mask = (ids >= lengths.unsqueeze(1)).bool()

    return mask # (batch, max_len)

