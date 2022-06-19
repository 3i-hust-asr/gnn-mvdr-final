from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import numpy as np
import torch
import nnet
import os

from scenario.train.util import *

def test_model(args):
    print('test model')
    model = nnet.get_model(args)
    nnet.print_summary(model, verbose=True)

    case = 4

    if case == 1:
        # n_filters = [64, 128, 128, 128, 16] ---- mask.py L23
        # mask_hidden = 256 ---- mvdr.py L17
        path = 'ckpt/mvdr/checkpoints/mvdr_epoch_0.ckpt'
        print(path, os.path.exists(path))
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])


    elif case == 2:
        # n_filters = [128, 128, 256, 128, 32] ---- mask.py L23
        # mask_hidden = 256 ---- mvdr.py L17
        path = 'ckpt/mvdr/checkpoints_mvdr_128/mvdr_epoch_0.ckpt'
        print(path, os.path.exists(path))
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])


    elif case == 3:
        # n_filters = [64, 128, 128, 128, 32] ---- mask.py L23
        # mask_hidden = 128 ---- mvdr.py L17
        path = 'ckpt/mvdr/checkpoints_mvdr_128_2/mvdr_epoch_0.ckpt'
        print(path, os.path.exists(path))
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])


    elif case == 4:
        # n_filters = [64, 128, 128, 128, 32] ---- mask.py L23
        # mask_hidden = 128 ---- mvdr.py L17
        path = 'ckpt/mvdr/checkpoints_mvdr_256/mvdr_epoch_0.ckpt'
        print(path, os.path.exists(path))
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
