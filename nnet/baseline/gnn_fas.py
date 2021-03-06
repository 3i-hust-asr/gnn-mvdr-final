from torch import nn
import torch
import math
import time

from ..common import Stft, GCN


class GNNFaS(nn.Module):

    def __init__(self, args, use_linear=True):
        super().__init__()
        self.n_filters = [64, 128, 256, 128, 32]
        self.kernel_size = 3
        self.stride = 2
        self.use_linear = use_linear

        chin = 8 

        self.stft = Stft(n_fft=1024, win_length=1024, hop_length=512, window='hann')
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for idx, n_filter in enumerate(self.n_filters):
            encode = [
                nn.Conv2d(chin, n_filter, kernel_size=self.kernel_size, stride=self.stride),
                nn.BatchNorm2d(num_features=n_filter),
                nn.SELU()
            ]
            self.encoder.append(nn.Sequential(*encode))
            decode = [
                nn.SELU(),
                nn.BatchNorm2d(num_features=n_filter),
                nn.ConvTranspose2d(n_filter, chin, kernel_size=self.kernel_size, stride=self.stride)
            ]
            self.decoder.insert(0, nn.Sequential(*decode))

            chin = n_filter

        linear_hidden = 160
        gcn_hidden = 64

        if self.use_linear:
            self.linear_1 = nn.Linear(linear_hidden, gcn_hidden)
            # self.gcn = GCN(gcn_hidden, gcn_hidden, args)
            self.gcn_1 = GCN(gcn_hidden, gcn_hidden, args)
            self.gcn_2 = GCN(gcn_hidden, gcn_hidden, args)
            self.linear_2 = nn.Linear(gcn_hidden, linear_hidden)
        else:
            # self.gcn = GCN(linear_hidden, linear_hidden, args)
            self.gcn_1 = GCN(linear_hidden, linear_hidden, args)
            self.gcn_2 = GCN(linear_hidden, linear_hidden, args)

    def valid_length(self, length):
        for idx in range(len(self.n_filters)):
            length = math.ceil((length - self.kernel_size) / self.stride) + 1
            length = max(length, 1)
        for idx in range(len(self.n_filters)):
            length = (length - 1) * self.stride + self.kernel_size
        return int(length)

    def forward(self, x, return_spec=False):
        # (B, L, C)
        B, L, C = x.shape
        t_length = torch.ones(B).int().fill_(L)

        x, _, f_length = self.stft(x, t_length)
        # (B, T, C, F, 2)
        # print('input  :', x.shape)

        r, i = x[..., 0], x[..., 1]
        # (B, T, C, F)

        frame = f_length[0]
        x = x.view(B, frame, C, -1).transpose(1, 2)
        freq = x.shape[-1] 
        # (B, C, T, 2F)

        # padding
        valid_T = self.valid_length(frame)
        valid_F = self.valid_length(freq)
        x = nn.functional.pad(x, (0, valid_F - freq, 0, valid_T - frame))
        # (B, C, valid_T, valid_F)
        # print('padded :', x.shape)

        # encoding
        skips = []
        for encode in self.encoder:
            x = encode(x)
            skips.append(x)
        # B, node, t, f
        # print('encoded:', x.shape)

        # GCN
        _, node, t, f = x.shape
        x = x.view(B, node, -1)
        # B, node, tf
        if self.use_linear:
            x = self.linear_1(x)
            # B, node, hidden
        # B, C, hidden
        gcn_1_out = self.gcn_1(x)
        gcn_2_out = self.gcn_2(gcn_1_out)
        # B, C, hidden
        x = x * gcn_2_out
        # B, node, hidden
        if self.use_linear:
            x = self.linear_2(x)
            # B, node, tf
        x = x.view(B, node, t, f)
        # B, node, t, f
        # print('gcn    :', x.shape)

        # decoding
        for decode in self.decoder:
            skip = skips.pop(-1)
            # x = torch.cat((x, skip), dim=-1)
            x = x + skip
            x = decode(x)
        # (B, C, valid_T, valid_F)
        # print('decoded:', x.shape)

        # cutoff
        x = x[:, :, :frame, :freq]
        # B, C, T, F
        # attention mask
        x = x.view(B, C, frame, freq//2, 2)
        # B, C, T, F, 2
        mask = x.mean(dim=1)
        # B, T, F, 2
        r_mask = mask[..., 0]
        i_mask = mask[..., 1]
        # B, T, F

        # reference channel = 0
        r_spec = r[:, :, 0]
        i_spec = i[:, :, 0]
        # (B, T, F)

        r_out_spec = r_mask * r_spec - i_mask * i_spec
        i_out_spec = r_mask * i_spec + i_mask * r_spec
        # B, T, F
        out_spec = torch.stack([r_out_spec, i_out_spec], dim=-1)
        # B, T, F, 2

        x, _ = self.stft.inverse(out_spec, t_length)
        # B, L

        if return_spec:
            return x, out_spec

        return x

    def forward_adj(self, x):
        # (B, L, C)
        B, L, C = x.shape
        t_length = torch.ones(B).int().fill_(L)

        x, _, f_length = self.stft(x, t_length)
        # (B, T, C, F, 2)
        # print('input  :', x.shape)

        r, i = x[..., 0], x[..., 1]
        # (B, T, C, F)

        frame = f_length[0]
        x = x.view(B, frame, C, -1).transpose(1, 2)
        freq = x.shape[-1] 
        # (B, C, T, 2F)

        # padding
        valid_T = self.valid_length(frame)
        valid_F = self.valid_length(freq)
        x = nn.functional.pad(x, (0, valid_F - freq, 0, valid_T - frame))
        # (B, C, valid_T, valid_F)
        # print('padded :', x.shape)

        # encoding
        skips = []
        for encode in self.encoder:
            x = encode(x)
            skips.append(x)
        # B, node, t, f
        # print('encoded:', x.shape)

        # GCN
        _, node, t, f = x.shape
        x = x.contiguous().view(B, node, -1)
        # B, node, tf
        if self.use_linear:
            x = self.linear_1(x)
            # B, node, hidden

        gcn_1_out, gcn_1_adj = self.gcn_1(x, return_adj=True)
        gcn_2_out, gcn_2_adj = self.gcn_2(gcn_1_out, return_adj=True)

        return gcn_1_adj, gcn_2_adj

    def encode(self, x):
        # (B, L, C)
        B, L, C = x.shape
        t_length = torch.ones(B).int().fill_(L)

        x, _, f_length = self.stft(x, t_length)
        # (B, T, C, F, 2)
        # print('input  :', x.shape)

        r, i = x[..., 0], x[..., 1]
        # (B, T, C, F)

        frame = f_length[0]
        x = x.view(B, frame, C, -1).transpose(1, 2)
        freq = x.shape[-1] 
        # (B, C, T, 2F)

        # padding
        valid_T = self.valid_length(frame)
        valid_F = self.valid_length(freq)
        x = nn.functional.pad(x, (0, valid_F - freq, 0, valid_T - frame))
        # (B, C, valid_T, valid_F)
        # print('padded :', x.shape)

        # encoding
        skips = []
        for encode in self.encoder:
            x = encode(x)
            skips.append(x)
        # B, node, t, f
        # print('encoded:', x.shape)

        # GCN
        _, node, t, f = x.shape
        x = x.contiguous().view(B, node, -1)
        # B, node, tf
        if self.use_linear:
            x = self.linear_1(x)
            # B, node, hidden
        z_in = x.clone()
        # B, C, hidden
        gcn_1_out = self.gcn_1(x)
        gcn_2_out = self.gcn_2(gcn_1_out)
        # B, C, hidden
        x = x * gcn_2_out
        # B, node, hidden
        if self.use_linear:
            x = self.linear_2(x)
            # B, node, tf
        x = x.view(B, node, t, f)
        # sB, node, t, f
        z_out = x.clone()
        # print('gcn    :', x.shape)
        return z_in, z_out

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