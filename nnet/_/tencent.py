import torch.nn as nn
import numpy as np
import torch

from .stft import Stft


class IPDFeature(nn.Module):
    '''
    Compute inter-channel phase difference
    '''
    def __init__(self,
                 ipd_index='0,0;1,0;2,0;3,0;4,0;5,0;6,0;7,0',
                 cos=True,
                 sin=False):
        super(IPDFeature, self).__init__()
        split_index = lambda sstr: [
            tuple(map(int, p.split(','))) for p in sstr.split(';')
        ]
        # ipd index
        pair = split_index(ipd_index)
        self.index_l = [t[0] for t in pair]
        self.index_r = [t[1] for t in pair]
        self.ipd_index = ipd_index
        self.cos = cos
        self.sin = sin
        self.num_pairs = len(pair) * 2 if cos and sin else len(pair)

    def extra_repr(self):
        return f'ipd_index={self.ipd_index}, cos={self.cos}, sin={self.sin}'

    def forward(self, p):
        '''
        Accept multi-channel phase and output inter-channel phase difference
        args
            p: phase matrix, N x C x F x T
        return
            ipd: N x MF x T
        '''
        if p.dim() not in [3, 4]:
            raise RuntimeError(
                '{} expect 3/4D tensor, but got {:d} instead'.format(
                    self.__name__, p.dim()))
        # C x F x T => 1 x C x F x T
        if p.dim() == 3:
            p = p.unsqueeze(0)
        N, _, _, T = p.shape
        pha_dif = p[:, self.index_l] - p[:, self.index_r]
        if self.cos:
            # N x M x F x T
            ipd = torch.cos(pha_dif)
            if self.sin:
                # N x M x 2F x T
                ipd = torch.cat([ipd, torch.sin(pha_dif)], 2)
        else:
            ipd = torch.fmod(pha_dif, 2 * math.pi) - math.pi
        # N x MF x T
        # ipd = ipd.contiguous().view(N, -1, T)
        # N x MF x T
        return ipd

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

    def forward(self, x):
        # (Batch, Channel, Sample)
        x = x.transpose(1, 2)

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
        return wav, None


    def compute_loss(self, mix, clean, noise):
        enhanced, _ = self(mix)
        loss = self.si_snr_loss(enhanced, clean)
        return {
            'loss': loss,
        }


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