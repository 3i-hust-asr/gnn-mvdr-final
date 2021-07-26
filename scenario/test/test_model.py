from torch.utils.tensorboard import SummaryWriter
from dgl.nn import EdgeWeightNorm, GraphConv
from torch_geometric.nn import GCNConv
import torch.nn as nn
import numpy as np
import torch
import dgl

import nnet




def test_model(args):
    print('test model')

    # g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    # g = dgl.add_self_loop(g)
    # print(g)
    # feat = torch.ones(6, 13)


    # edge_weight = torch.tensor([0.5, 0.6, 0.4, 0.7, 0.9, 0.1, 0.5, 0.6, 0.4, 0.7, 0.9, 0.1])
    # print(edge_weight.shape)
    # norm = EdgeWeightNorm(norm='both')
    # norm_edge_weight = norm(g, edge_weight)
    # us, vs = g.all_edges()
    # print([(u.item(),v.item()) for (u,v) in zip(us,vs)])
    # print(norm_edge_weight)
    # print(norm_edge_weight.shape)

    # conv = GraphConv(13, 2, norm='none', weight=True, bias=True)
    # res = conv(g, feat, edge_weight=norm_edge_weight)
    # print(res.shape)

    model = nnet.get_model(args)
    nnet.print_summary(model)
    x = torch.randn(3, 96000, 8)
    loss = model.compute_loss(x, x[...,0])
    print(loss)

    # debug
    # writer = SummaryWriter('logs')
    # writer.add_graph(model, x)
    # writer.close()

    # conv = GCNConv(11, 13)
    # edge_index = torch.tensor([[0,1,2,3,2,5], [1,2,3,4,0,3]], dtype=torch.long)
    # edge_weight = torch.randn(6, 3)
    # print(edge_weight.shape)
    # batch_edge = torch.stack((edge_index, edge_index, edge_index), dim=0)
    # print(batch_edge.shape)
    # x = conv(x, edge_index, edge_weight)
    # print(x.shape)


    # gcn = GCN(input_dim=11, output_dim=13)
    # x = torch.ones(2, 6, 11)

    # x = gcn(x)
    # print(x.shape)