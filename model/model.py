import torch
import torch.nn as nn

from module import Encoder, Decoder, PostNet
from utils import get_mask_from_lengths


class MyModel(nn.Module):
    def __init__(self, configs):
        super().__init__()
        preprocess_cfg, model_cfg, train_cfg = configs
        
        self.encoder = Encoder(model_cfg)
        self.decoder = Decoder(preprocess_cfg, model_cfg)
        self.postnet = PostNet()


    def forward(
        self,

        text,
        text_lens,
        mel,
        gate,
        mel_lens
    ):
        '''
         text: (batch, max_text_len)
         text_lens: (batch)
         mel: (batch, 80, max_mel_len)
         gate: (batch, max_mel_len)
         mel_lens: (batch)
        '''
        
        memory, processed_memory = self.encoder(text, text_lens)

        mel_pred, gate_pred, alignment = self.decoder(memory, processed_memory, text_lens, mel)

        mel_pred_postnet = self.postnet(mel_pred) + mel_pred
        '''
        mask = get_mask_from_lengths(mel_lens) # (batch, max_mel_len)
        mel_pred = mel_pred.masked_fill(mask.unsqueeze(1), 0.0)
        mel_pred_postnet = mel_pred_postnet.masked_fill(mask.unsqueeze(1), 0.0)
        gate_pred = gate_pred.masked_fill(mask, 1e3)
        '''
        return mel_pred, mel_pred_postnet, gate_pred, alignment


    def inference(
        self,

        text
    ):
        '''
         text: (batch, max_text_len)
        '''
        
        memory, processed_memory = self.encoder.inference(text)

        mel_pred, gate_pred, alignment = self.decoder.inference(memory, processed_memory)

        mel_pred_postnet = self.postnet(mel_pred) + mel_pred
        
        return mel_pred, mel_pred_postnet, gate_pred, alignment
