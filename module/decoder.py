import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np

from .prenet import Prenet
from .lsa import LSA
from .layers import LinearNorm
from utils import get_mask_from_lengths


class Decoder(nn.Module):
    """ Decoder """

    def __init__(self, preprocess_cfg, model_cfg):
        super(Decoder, self).__init__()
        
        self.n_mel_channels = preprocess_cfg['mel']['n_mel_channels']
        self.n_frames_per_step = model_cfg['decoder']['n_frames_per_step']
        self.encoder_embedding_dim = model_cfg['encoder']['encoder_embedding_dim']
        self.attention_rnn_dim = model_cfg['decoder']['attention_rnn_dim']
        self.decoder_rnn_dim = model_cfg['decoder']['decoder_rnn_dim']
        self.prenet_dim = model_cfg['decoder']['prenet_dim']
        self.attention_rnn_dropout = model_cfg['decoder']['attention_rnn_dropout']
        self.decoder_rnn_dropout = model_cfg['decoder']['decoder_rnn_dropout']
        attention_dim = model_cfg["decoder"]["attention_dim"]
        attention_location_n_filters = model_cfg['decoder']['attention_location_n_filters']
        attention_location_kernel_size = model_cfg['decoder']['attention_location_kernel_size']

        self.prenet = Prenet(
            self.n_mel_channels * self.n_frames_per_step,
            [self.prenet_dim, self.prenet_dim])

        self.attention_rnn = nn.LSTMCell(
            self.prenet_dim + self.encoder_embedding_dim,
            self.attention_rnn_dim)

        self.decoder_rnn = nn.LSTMCell(
            self.attention_rnn_dim + self.encoder_embedding_dim,
            self.decoder_rnn_dim)

        self.attention = LSA(
            self.attention_rnn_dim, self.encoder_embedding_dim,
            attention_dim, attention_location_n_filters,
            attention_location_kernel_size)

        self.mel_linear = LinearNorm(
            self.decoder_rnn_dim + self.encoder_embedding_dim,
            self.n_mel_channels * self.n_frames_per_step)

        self.gate_linear = LinearNorm(
            self.decoder_rnn_dim + self.encoder_embedding_dim,
            1, w_init_gain='sigmoid')

    def inference(self, memory, processed_memory):
        '''
         memory(encoder_output): (batch, max_text_len, encoder_d)
         memory_lens: (batch)
        '''
        frame = self.prenet(self.get_go_frame(memory))

        self.initialize_decoder_states(memory, processed_memory)

        mel_outputs, gate_outputs, alignments = [], [], []
        while True:
            mel_output, gate_output, attention_weights = self.decode(frame)
            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output.squeeze(1)]
            alignments += [attention_weights]
            if torch.sigmoid(gate_output.data) > 0.5:
                break
            elif len(mel_outputs) == 1000:
                print("Warning! Reached max decoder steps")
                break
            frame = self.prenet(mel_output)

        mel_outputs, gate_outputs, alignments = self.parse_outputs(
            mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments

    def forward(self, memory, processed_memory, memory_lens, mel_truth):
        '''
         memory(encoder_output): (batch, max_text_len, encoder_d)
         memory_lens: (batch)
         mel_truth: (batch, 80, max_mel_len)
        '''
        go_frame = self.get_go_frame(memory).unsqueeze(0) # (1, batch, 80*1)
        truth = self.parse_mel_truth(mel_truth) # (max_mel_len/1, batch, 80*1)
        truth = torch.cat((go_frame, truth), dim=0) # (max_mel_len/1+1, batch, 80*1)
        truth = self.prenet(truth) # (max_mel_len/1+1, batch, 256)

        self.initialize_decoder_states(memory, processed_memory, memory_lens)

        mel_outputs, gate_outputs, alignments = [], [], []
        while len(mel_outputs) < truth.size(0) - 1:
            frame = truth[len(mel_outputs)]
            mel_output, gate_output, attention_weights = self.decode(frame)
            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output.squeeze(1)]
            alignments += [attention_weights]

        mel_outputs, gate_outputs, alignments = self.parse_outputs(
            mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments

    def decode(self, frames):
        '''Decoder step using stored states, attention and memory
        -----
         frames: (n_frames, batch, 256)
        '''
        # attention_rnn
        attention_rnn_input = torch.cat((frames, self.attention_context), -1)
        self.attention_rnn_h, self.attention_rnn_c = self.attention_rnn(
            attention_rnn_input, (self.attention_rnn_h, self.attention_rnn_c))
        self.attention_rnn_h = F.dropout(
            self.attention_rnn_h, self.attention_rnn_dropout, self.training)

        # LSA
        attention_weights_cat = torch.cat(
            (self.attention_weights.unsqueeze(1),
             self.attention_weights_cum.unsqueeze(1)), dim=1)
        self.attention_context, self.attention_weights = self.attention(
            self.attention_rnn_h, self.memory, self.processed_memory,
            attention_weights_cat, self.memory_mask)
        self.attention_weights_cum += self.attention_weights

        # decoder_rnn
        decoder_rnn_input = torch.cat(
            (self.attention_rnn_h, self.attention_context), -1)
        self.decoder_rnn_h, self.decoder_rnn_c = self.decoder_rnn(
            decoder_rnn_input, (self.decoder_rnn_h, self.decoder_rnn_c))
        self.decoder_rnn_h = F.dropout(
            self.decoder_rnn_h, self.decoder_rnn_dropout, self.training)

        # cat
        decoder_rnn_h_attention_context = torch.cat(
            (self.decoder_rnn_h, self.attention_context), dim=1)

        # mel_output
        mel_output = self.mel_linear(decoder_rnn_h_attention_context)

        # gate_output
        gate_output = self.gate_linear(decoder_rnn_h_attention_context)

        return mel_output, gate_output, self.attention_weights

    def get_go_frame(self, memory):
        """ Gets all zeros frames to use as first decoder input
         memory: encoder outputs (batch, max_text_len, encoder_d)
         return: all zeros frames (batch, 80*1)
        """
        B = memory.size(0)
        go_frame = Variable(memory.data.new(
            B, self.n_mel_channels * self.n_frames_per_step).zero_())
        return go_frame

    def parse_mel_truth(self, mel_truth):
        """ 
         mel_truth: used for teacher-forced training
         return: processed mel_truth
        """
        # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)
        mel_truth = mel_truth.transpose(1, 2)
        truth = mel_truth.view(
            mel_truth.size(0),
            int(mel_truth.size(1)/self.n_frames_per_step), -1)
        # (B, T_out, n_mel_channels) -> (T_out, B, n_mel_channels)
        truth = truth.transpose(0, 1)
        return truth

    def parse_outputs(self, mel_outputs, gate_outputs, alignments):
        # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        # decouple frames per step
        mel_outputs = mel_outputs.view(
            mel_outputs.size(0), -1, self.n_mel_channels)
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)

        # (T_out, B) -> (B, T_out)
        gate_outputs = torch.stack(gate_outputs).transpose(0, 1)
        gate_outputs = gate_outputs.contiguous()

        # (T_out, B) -> (B, T_out)
        alignments = torch.stack(alignments).transpose(0, 1)
        
        return mel_outputs, gate_outputs, alignments

    def initialize_decoder_states(self, memory, processed_memory, memory_lens=None):
        """ Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory
        ------
         memory(encoder_output): (batch, max_text_len, encoder_d)
         memory_lens: (batch, max_text_len)
        """
        B = memory.size(0)
        max_text_len = memory.size(1)

        self.attention_rnn_h = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())
        self.attention_rnn_c = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())

        self.decoder_rnn_h = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())
        self.decoder_rnn_c = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())

        self.attention_weights = Variable(memory.data.new(
            B, max_text_len).zero_())
        self.attention_weights_cum = Variable(memory.data.new(
            B, max_text_len).zero_())
        self.attention_context = Variable(memory.data.new(
            B, self.encoder_embedding_dim).zero_())

        self.memory = memory
        self.processed_memory = processed_memory
        self.memory_mask = None
        if memory_lens is not None:
            self.memory_mask = get_mask_from_lengths(memory_lens) # (batch, max_text_len)
    
    
