from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import numpy as np
import torch
import nnet

def test_model(args):
    print('test model')
    model = nnet.get_model(args)
    nnet.print_summary(model)
    x = torch.randn(3, 16000, 8).to(args.device)
    loss = model.compute_loss(x, x[..., 0])
    print(loss)

