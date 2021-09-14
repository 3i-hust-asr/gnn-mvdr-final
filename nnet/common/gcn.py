from torch import nn
import torch

class GCN(nn.Module):

    def __init__(self, input_dim, output_dim, args, bias=True):
        super(GCN, self).__init__()
        self.adj_transform = nn.Linear(input_dim*2, 1)

        weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        torch.nn.init.xavier_normal_(weight)
        self.register_parameter('weight', weight)

        bias = nn.Parameter(torch.Tensor(output_dim))
        torch.nn.init.normal_(bias)
        self.register_parameter('bias', bias)

    def forward(self, x, return_adj=False):
        B, N, F = x.shape
        # construct adjacency matrix
        tmp = x.unsqueeze(1).expand(-1, N, -1, -1)
        adj = torch.cat((tmp, tmp.transpose(1, 2)), dim=-1)

        # non-linear function
        adj = self.adj_transform(adj).squeeze(-1) # (B, N, N)

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
        if return_adj:
            return out, adj
        return out
