from torch_complex.tensor import ComplexTensor
from torch_complex import functional as FC
from typing import Tuple, Union, List
from torch.nn import functional as F
import torch.nn as nn
import torch

from ..common import Stft
from .util import *

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

        self.ref_channel = 0

    def forward(self, x, return_spec=False):
        # (Batch, Sample, Channel)
        B, L, C = x.shape
        t_length = torch.ones(B).int().fill_(L)
        spectrum, phase, f_length = self.stft(x, t_length)
        # (Batch, Frame, Channel, Frequency, 2)

        ################################ Feature ################################
        r, i = spectrum[..., 0], spectrum[..., 1]
        # (Batch, Frame, Channel, Frequency)
        r_spec = r.transpose(1, 2)
        i_spec = i.transpose(1, 2)
        # (Batch, Channel, Frame, Frequency)
        x = torch.cat([r_spec, i_spec], dim=-1)
        # (Batch, Channel, Frame, Frequency*2)
        x  = x.view(-1, x.shape[2], x.shape[3])
        # (Batch*Channel, Frame, Frequency*2)

        ################################ Mask ################################
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

        ################################ Beamforming ################################
        enhance, _ = apply_mvdr(spectrum, speech_spec, noise_spec, self.ref_channel)
        # (Batch, Frequency, Frame)
        enhance = enhance.transpose(1, 2)
        # (Batch, Frame, Frequency)
        enhance = torch.stack([enhance.real, enhance.imag], dim=-1)
        # (Batch, Frame, Frequency, 2)
        wav, _ = self.stft.inverse(enhance, t_length)
        # (Batch, Sample)
        if return_spec:
            return wav, enhance
        return wav

    def compute_loss(self, mix, clean):
        enhanced_signal, enhanced_spec = self(mix, return_spec=True)
        clean_spec = self.stft(clean)[0]
        loss_mag = torch.view_as_complex(clean_spec - enhanced_spec).abs().mean()
        loss_raw = (clean - enhanced_signal).abs().mean()
        loss_sisnr = self.si_snr_loss(enhanced_signal, clean)
        loss_sisnr = 0 #self.si_snr_loss(enhanced_signal, clean)
        return loss_raw + loss_mag + loss_sisnr

    def si_snr_loss(self, ref, inf):
        """si-snr loss
            :param ref: (Batch, samples)
            :param inf: (Batch, samples)
            :return: (Batch)
        """
        ref = ref / torch.norm(ref, p=2, dim=1, keepdim=True)
        inf = inf / torch.norm(inf, p=2, dim=1, keepdim=True)

        s_target = (ref * inf).sum(dim=1, keepdims=True) * ref
        e_noise = inf - s_target

        si_snr = 20 * torch.log10(
            torch.norm(s_target, p=2, dim=1) / torch.norm(e_noise, p=2, dim=1)
        )
        return -si_snr.mean()
    