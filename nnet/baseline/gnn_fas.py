import torch.nn.functional as F
from torch import nn
import torch
import math

from ..common import Stft


class GCN(nn.Module):

    def __init__(self, input_dim, output_dim, args, bias=True):
        super(GCN, self).__init__()
        self.adj_transform = nn.Linear(input_dim*2, 1)

        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim)).to(args.device)
        torch.nn.init.xavier_normal_(self.weight)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim)).to(args.device)
            torch.nn.init.normal_(self.bias)
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        B, N, F = x.shape
        # construct adjacency matrix
        adj = torch.zeros((B, N, N, 2 * F), device=x.device)
        for i in range(N):
            for j in range(N):
                adj[:, i, j, :] = torch.cat((x[:, i, :], x[:, j, :]), dim=-1)
        # non-linear function
        adj = self.adj_transform(adj).squeeze(-1) # (B, N, N)
        # normalize
        # adj = adj.softmax(dim=1)
        # add self loop
        idx = torch.arange(N, dtype=torch.long, device=x.device)
        adj[:, idx, idx] += 1.0
        # convolution
        out = torch.matmul(x, self.weight)
        deg_inv_sqrt = adj.sum(dim=-1).clamp(min=1).pow(-0.5)
        adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)
        out = torch.matmul(adj, out)
        # add bias
        if self.bias is not None:
            out = out + self.bias
        # activation
        out = out.relu()
        return out


class GNNFaS(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.n_filters = [64, 128, 128, 256, 256, 256]
        self.kernel_size = 3
        self.stride = 2

        chin = 2 # complex

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

        hidden = 512
        self.gcn = GCN(hidden, hidden, args)

        self.linear_1 = nn.Linear(4096, hidden)
        self.linear_2 = nn.Linear(hidden, 4096)


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

        x, _, f_length = self.stft(x, t_length)
        # (B, T, C, F, 2)

        r, i = x[..., 0], x[..., 1]
        # (B, T, C, F)

        x = x.permute(0, 2, 4, 1, 3)
        # (B, C, 2, T, F)

        frame = f_length[0]
        freq = x.shape[-1] 
        x = x.view(B*C, 2, frame, freq)
        # (B*C, 2, T, F)
        # print('input  :', x.shape)

        # padding
        valid_T = self.valid_length(frame)
        valid_F = self.valid_length(freq)
        x = F.pad(x, (0, valid_F - freq, 0, valid_T - frame))
        # (B*C, 2, valid_T, valid_F)
        # print('padded :', x.shape)

        # encoding
        skips = []
        for encode in self.encoder:
            x = encode(x)
            skips.append(x)
        # B*C, filter[-1], t, f

        # GCN
        _, h, t, f = x.shape
        x = x.contiguous().view(B, C, -1)
        # x = x.reshape(B, C, -1)
        # B, C, 4096
        x = self.linear_1(x)
        # B, C, hidden
        gcn_out = self.gcn(x)
        # B, C, hidden
        x = x * gcn_out
        # B, C, hidden
        x = self.linear_2(x)
        # B, C, 4096
        x = x.view(B*C, h, t, f)
        # B*C, filter[-1], t, f
        # print('gcn    :', x.shape)

        # decoding
        for decode in self.decoder:
            skip = skips.pop(-1)
            # x = torch.cat((x, skip), dim=-1)
            x = x + skip
            x = decode(x)
        # B*C, 2, T, F
        # print('decoded:', x.shape)

        # attention mask
        x = x.view(B, C, 2, valid_T, valid_F)
        # B, C, 2, valid_T, valid_F
        mask = x.mean(dim=1)
        # B, 2, valid_T, valid_F
        r_mask = mask[:, 0, :frame, :freq]
        i_mask = mask[:, 1, :frame, :freq]
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
        return x


    def compute_loss(self, mix, clean):
        enhanced = self(mix)
        return self.si_snr_loss(enhanced, clean)


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