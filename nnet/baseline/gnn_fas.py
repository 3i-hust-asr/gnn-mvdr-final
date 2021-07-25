import torch.nn.functional as F
from torch import nn
import torch
import math

from ..common import Stft

class GNNFaS(nn.Module):

    def __init__(self):
        super().__init__()
        self.n_filters = [64, 128, 128, 256, 256, 256]
        self.kernel_size = 3
        self.stride = 2
        chin = 8
        chout = 8

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


    def valid_length(self, length):
        for idx in range(len(self.n_filters)):
            length = math.ceil((length - self.kernel_size) / self.stride) + 1
            length = max(length, 1)
        for idx in range(len(self.n_filters)):
            length = (length - 1) * self.stride + self.kernel_size
        return int(length)

    def forward(self, x):
        # (B, L, C)
        B, L, C = x.shape
        t_length = torch.ones(B).int().fill_(L)

        spectrum, phase, f_length = self.stft(x, t_length)
        # spectrum: (B, T, C, F, 2), phase: (B, T, C, F)

        frame = f_length[0]
        x = spectrum.view(B, frame, C, -1).permute(0, 2, 1, 3)
        # (B, C, T, F*2)
        freq = x.shape[-1] 

        # print('valid F:', self.valid_length(freq))
        # print('valid T:', self.valid_length(frame))

        x = F.pad(x, (0, self.valid_length(freq) - freq, 
                      0, self.valid_length(frame) - frame))
        # (B, C, valid_T, valid_F)
        print('input  :', x.shape)

        skips = []
        for encode in self.encoder:
            x = encode(x)
            skips.append(x)

        print('encoded:', x.shape)

        for decode in self.decoder:
            skip = skips.pop(-1)
            # x = torch.cat((x, skip), dim=-1)
            x = x + skip
            x = decode(x)

        print('decoded:', x.shape)

        # print(type(spectrum), spectrum.dtype, spectrum.shape)

        return spectrum