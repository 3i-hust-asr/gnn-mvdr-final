from torch.utils.tensorboard import SummaryWriter
from dgl.nn import EdgeWeightNorm, GraphConv
import numpy as np
import torch
import dgl

import nnet

def test_model(args):
    print('test model')

    # g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    # print(g)
    # g = dgl.add_self_loop(g)
    # feat = torch.ones(6, 13)
    # edge_weight = torch.tensor([0.5, 0.6, 0.4, 0.7, 0.9, 0.1, 0.5, 0.6, 0.4, 0.7, 0.9, 0.1])
    # print(edge_weight.shape)
    # norm = EdgeWeightNorm(norm='both')
    # norm_edge_weight = norm(g, edge_weight)
    # print(norm_edge_weight.shape)
    # conv = GraphConv(13, 2, norm='none', weight=True, bias=True)
    # res = conv(g, feat, edge_weight=norm_edge_weight)
    # print(res.shape)

    model = nnet.get_model(args)
    print(model)
    x = torch.randn(3, 96000, 8)
    model(x)

    # debug
    # writer = SummaryWriter('logs')
    # writer.add_graph(model, x)
    # writer.close()
