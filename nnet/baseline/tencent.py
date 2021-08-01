import torch.nn as nn
import numpy as np
import torch

from ..common import IPDFeature
from ..common import Stft

class Tencent(nn.Module):

    def __init__(self,
                 fft_len=512,
                 win_len=320,
                 win_inc=160,
                 win_type='hann',
                 rnn_units=512,
                 rnn_layers=3,
                 bidirectional=False,
                 ipd_index='0,4;1,5;2,6;3,7',
                 cos=True,
                 sin=False,
                ):
        super(Tencent, self).__init__()

        num_bins = fft_len // 2 + 1
        ipd_num = len(ipd_index.split(';'))

        self.ipd_extractor = IPDFeature(ipd_index=ipd_index,
                                        cos=cos,
                                        sin=sin)
        
        self.stft = Stft(n_fft=fft_len,
                         win_length=win_len,
                         hop_length=win_inc,
                         window=win_type)
        
        self.lstm = nn.LSTM(input_size=(2+ipd_num)*num_bins,
                            hidden_size=rnn_units,
                            num_layers=rnn_layers,
                            bidirectional=bidirectional,
                            batch_first=True)

        fac = 2 if bidirectional else 1
        self.linear = nn.Linear(rnn_units * fac, num_bins * 2)

    def forward(self, x, return_spec=False):
        # (Batch, Sample, Channel)
        B, L, C = x.shape
        t_length = torch.ones(B).int().fill_(L)

        spectrum, phase, f_length = self.stft(x, t_length)
        # spectrum: (Batch, Frame, Channel, Frequency, 2)
        # phase: (Batch, Frame, Channel, Frequency)

        phase = phase.permute(0, 2, 3, 1)
        # (Batch, Channel, Frequency, Frame)

        ipd = self.ipd_extractor(phase)
        # (Batch, ipd_num, Frequency, Frame)

        ipd = ipd.contiguous().view(B, -1, f_length[0])
        # (Batch, ipd_num * Frequency, Frame)

        ipd = ipd.transpose(1, 2)
        # (Batch, Frame, ipd_num * Frequency)

        r, i = spectrum[..., 0], spectrum[..., 1]
        # (Batch, Frame, Channel, Frequency)

        r_spec = r[:, :, 0]
        i_spec = i[:, :, 0]
        # (Batch, Frame, Frequency)

        inp = torch.cat([r_spec, i_spec, ipd], dim=-1)
        # (Batch, Frame, (ipd_num+2)*Frequency)

        out, _ = self.lstm(inp)
        # (Batch, Frame, Hidden)

        mask = self.linear(out)
        # (Batch, Frame, 2 * Frequency)

        r_mask, i_mask = torch.chunk(mask, 2, 2)
        # (Batch, Frame, Frequency)

        r_out_spec = r_mask * r_spec - i_mask * i_spec
        i_out_spec = r_mask * i_spec + i_mask * r_spec
        # (Batch, Frame, Frequency)

        out_spec = torch.stack([r_out_spec, i_out_spec], dim=-1)
        # (Batch, Frame, Frequency, 2)

        wav, _ = self.stft.inverse(out_spec, t_length)
        # (Batch, Sample)
        if return_spec:
            return wav, out_spec
        return wav


    # def compute_loss(self, mix, clean):
    #     enhanced = self(mix)
    #     loss = self.si_snr_loss(enhanced, clean)
    #     return loss

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