import torch.nn as nn
import torch

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