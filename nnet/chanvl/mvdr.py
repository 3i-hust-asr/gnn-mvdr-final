from torch_complex.tensor import ComplexTensor
from torch_complex import functional as FC
from typing import Tuple, Union, List
from torch.nn import functional as F
import torch.nn as nn
import torch

from ..common import Stft

class Mvdr(torch.nn.Module):

    def __init__(self):
        super(Mvdr, self).__init__()

        fft_len=512
        rnn_units=128
        bidirectional = False

        num_bins = fft_len // 2 + 1

        self.stft = Stft(n_fft=fft_len, win_length=320, hop_length=160, window='hann')
        self.lstm = nn.LSTM(input_size=num_bins*2, hidden_size=rnn_units, num_layers=2, bidirectional=bidirectional, batch_first=True)

        fac = 2 if bidirectional else 1
        self.linears = torch.nn.ModuleList([torch.nn.Linear(rnn_units * fac, num_bins * 2) for _ in range(2)])

    def forward(self, x):
        # (Batch, Sample, Channel)
        B, L, C = x.shape
        t_length = torch.ones(B).int().fill_(L)

        spectrum, phase, f_length = self.stft(x, t_length)
        # spectrum: (Batch, Frame, Channel, Frequency, 2)

        r, i = spectrum[..., 0], spectrum[..., 1]
        # (Batch, Frame, Channel, Frequency)

        r_spec = r.transpose(1, 2)
        i_spec = i.transpose(1, 2)
        # (Batch, Channel, Frame, Frequency)

        x = torch.cat([r_spec, i_spec], dim=-1)
        # (Batch, Channel, Frame, Frequency*2)

        x  = x.view(-1, x.shape[2], x.shape[3])
        # (Batch*Channel, Frame, Frequency*2)

        x, _ = self.lstm(x)
        # (Batch*Channel, Frame, Hidden)

        specs = []
        for linear in self.linears:
            mask = linear(x)
            # (Batch*Channel, Frame, 2 * Frequency)

            r_mask, i_mask = torch.chunk(mask, 2, 2)
            # (Batch*Channel, Frame, Frequency)

            r_mask = r_mask.view(B, C, r_mask.shape[1], r_mask.shape[2])
            i_mask = i_mask.view(B, C, i_mask.shape[1], i_mask.shape[2])
            # (Batch, Channel, Frame, Frequency)

            r_out_spec = r_mask * r_spec - i_mask * i_spec
            i_out_spec = r_mask * i_spec + i_mask * r_spec
            # (Batch, Channel, Frame, Frequency)

            specs.append(torch.stack([r_out_spec, i_out_spec], dim=-1))

        speech_spec, noise_spec = specs[0], specs[1]
        # (Batch, Channel, Frame, Frequency, 2)
