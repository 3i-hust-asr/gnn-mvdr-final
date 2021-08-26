from torch import nn
import torch

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

    def forward(self, x, return_adj=False):
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
        if return_adj:
            return out, adj
        return out
