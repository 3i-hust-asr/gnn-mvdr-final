import torch.nn as nn
import torch
import math

class MaskLstm(torch.nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.module = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=2, batch_first=True)

    def forward(self, x):
        # (Batch, Channel, Frame, Frequency*2)
        x  = x.view(-1, x.shape[2], x.shape[3])
        # (Batch*Channel, Frame, Frequency*2)
        x, _ = self.module(x)
        # (Batch*Channel, Frame, Hidden)
        return x

class MaskGNNUnet(torch.nn.Module):

    def __init__(self, embed_class, args, input_size=None, hidden_size=None, use_linear=True):
        super().__init__()
        self.n_filters = [128, 128, 256, 128, 32]
        self.kernel_size = 3
        self.stride = 2
        self.use_linear = use_linear
        self.embed_class = embed_class
        chin = 8 

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

        linear_hidden = 288
        embed_hidden = 64

        if self.use_linear:
            self.linear_1 = nn.Linear(linear_hidden, embed_hidden)
            self.embed_1 = self.embed_class(embed_hidden, embed_hidden, args)
            self.embed_2 = self.embed_class(embed_hidden, embed_hidden, args)
            self.linear_2 = nn.Linear(embed_hidden, linear_hidden)
        else:
            self.embed_1 = self.embed_class(linear_hidden, linear_hidden, args)
            self.embed_2 = self.embed_class(linear_hidden, linear_hidden, args)

        self.output = nn.Linear(input_size, hidden_size)

    def valid_length(self, length):
        for idx in range(len(self.n_filters)):
            length = math.ceil((length - self.kernel_size) / self.stride) + 1
            length = max(length, 1)
        for idx in range(len(self.n_filters)):
            length = (length - 1) * self.stride + self.kernel_size
        return int(length)

    def forward(self, x):
        B, C, T, F = x.shape
        valid_T = self.valid_length(T)
        valid_F = self.valid_length(F)
        x = nn.functional.pad(x, (0, valid_F - F, 0, valid_T - T))

        # encoding
        skips = []
        for encode in self.encoder:
            x = encode(x)
            skips.append(x)

        # gcn
        _, node, t, f = x.shape
        x = x.contiguous().view(B, node, t*f)
        if self.use_linear:
            x = self.linear_1(x)

        out_1 = self.embed_1(x)
        out_2 = self.embed_2(out_1)
        x = x * out_2
        
        if self.use_linear:
            x = self.linear_2(x)
        x = x.view(B, node, t, f)

        # decoding
        for decode in self.decoder:
            skip = skips.pop(-1)
            x = x + skip
            x = decode(x)
        
        # cutoff
        x = x[:, :, :T, :F]
        # B, C, T, F
        x = x.view(B*C, T, F)
        return self.output(x)
