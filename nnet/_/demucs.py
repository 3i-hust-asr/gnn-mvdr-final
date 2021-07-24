import torch.nn.functional as F
from torch import nn
import julius

from .util import *
from .loss import MultiResolutionSTFTLoss

class BLSTM(nn.Module):
    
    def __init__(self, dim, layers=2, bi=True):
        super().__init__()
        self.lstm = nn.LSTM(
            bidirectional=bi,
            num_layers=layers,
            hidden_size=dim,
            input_size=dim,
        )
        self.linear = nn.Linear(2 * dim, dim) if bi else nn.Linear(dim, dim)

    def forward(self, x, hidden=None):
        x, hidden = self.lstm(x, hidden)
        x = self.linear(x)
        return x, hidden

class Demucs(nn.Module):
    
    'Demucs speech enhancement model'

    def __init__(
        self,
        chin=1,
        chout=1,
        ref_channel=0,
        hidden=32,
        depth=3,
        kernel_size=8,
        stride=4,
        causal=True,
        resample=2,
        growth=2,
        max_hidden=10000,
        normalize=True,
        glu=False,
        rescale=0.1,
        floor=1e-7,
    ):

        super().__init__()
        if resample not in [1, 2, 4]:
            raise ValueError("Resample should be 1, 2 or 4.")

        self.chin = chin
        self.chout = chout
        self.hidden = hidden
        self.depth = depth
        self.kernel_size = kernel_size
        self.stride = stride
        self.causal = causal
        self.floor = floor
        self.resample = resample
        self.normalize = normalize
        self.ref_channel = ref_channel

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        activation = nn.GLU(1) if glu else nn.ReLU()
        ch_scale = 2 if glu else 1

        self.criterion_1 = F.smooth_l1_loss
        self.criterion_2 = MultiResolutionSTFTLoss()

        for index in range(depth):
            encode = [
                nn.Conv1d(chin, hidden, kernel_size, stride),
                nn.ReLU(),
                nn.Conv1d(hidden, hidden * ch_scale, 1),
                activation,
            ]
            self.encoder.append(nn.Sequential(*encode))

            decode = [
                nn.Conv1d(hidden, ch_scale * hidden, 1),
                activation,
                nn.ConvTranspose1d(hidden, chout, kernel_size, stride),
            ]
            if index > 0:
                decode.append(nn.ReLU())
            self.decoder.insert(0, nn.Sequential(*decode))
            chout = hidden
            chin = hidden
            hidden = min(int(growth * hidden), max_hidden)

        self.lstm = BLSTM(chin, bi=not causal)
        if rescale:
            rescale_module(self, reference=rescale)

    def valid_length(self, length):
        length = math.ceil(length * self.resample)
        for idx in range(self.depth):
            length = (
                math.ceil((length - self.kernel_size) / self.stride)
                + 1
            )
            length = max(length, 1)
        for idx in range(self.depth):
            length = (length - 1) * self.stride + self.kernel_size
        length = int(math.ceil(length / self.resample))
        return int(length)

    @property
    def total_stride(self):
        return self.stride ** self.depth // self.resample

    def forward(self, mix):
        # B, C, L = mix.shape
        mix = mix[:, self.ref_channel, :]
        mix = mix.unsqueeze(1)

        if self.normalize:
            mono = mix.mean(dim=1, keepdim=True)
            std = mono.std(dim=-1, keepdim=True)
            mix = mix / (self.floor + std)
        else:
            std = 1

        length = mix.shape[-1]
        x = mix
        x = F.pad(x, (0, self.valid_length(length) - length))

        if self.resample == 2:
            x = upsample2(x) 
        elif self.resample == 4:  
            x = upsample2(x)
            x = upsample2(x)

        skips = []
        for encode in self.encoder:
            x = encode(x)
            skips.append(x)

        x = x.permute(2, 0, 1)
        x, _ = self.lstm(x)
        x = x.permute(1, 2, 0)

        for decode in self.decoder:
            skip = skips.pop(-1)
            x = x + skip[..., : x.shape[-1]]
            x = decode(x)
            
        if self.resample == 2:
            x = downsample2(x)
        elif self.resample == 4:
            x = downsample2(x)
            x = downsample2(x)

        x = x[..., :length]
        return (std * x).squeeze(1), None


    def compute_loss(self, mix, clean, noise):
        enhanced, _ = self(mix)
        # loss = self.criterion_1(enhanced, clean)
        # sc_loss, mag_loss = self.criterion_2(enhanced, clean)
        # loss += (sc_loss + mag_loss)
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
        return -si_snr