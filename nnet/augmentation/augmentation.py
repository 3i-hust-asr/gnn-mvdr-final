import numpy as np
import torch
import time

from .util import *

class Augmentation(torch.nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, cleans, noises, rirs):
        # extract parameters
        args = self.args
        batch_size = len(rirs)
        # convert to device
        cleans = cleans.to(args.device)
        noises = noises.to(args.device)
        # extract rirs
        clean_rirs, noise_rirs = extract_rirs(rirs, args)
        # add reverb to cleans
        clean_reverbs = torch.stack([conv1d(cleans[i], clean_rirs[i]) for i in range(batch_size)])
        # add reverb to noises
        noise_reverbs = torch.stack([conv1d(noises[i], noise_rirs[i]) for i in range(batch_size)])
        # mix
        # snr_range=[0,10], scale_range=[0.2,0.9]
        # snr = np.random.rand() * 10
        # snr = 5
        snr = np.random.choice([-7.5, -5, 0, 5, 7.5])
        # scale = np.random.rand() * 0.7 + 0.2
        scale = 0.5
        
        inputs, clean_reverbs, noise_reverbs = mix(clean_reverbs, noise_reverbs, args, snr=snr, scale=scale)

        # transpose
        inputs        = inputs.transpose(1, 2)
        clean_reverbs = clean_reverbs.transpose(1, 2)
        noise_reverbs = noise_reverbs.transpose(1, 2)
        return inputs, cleans, noises, clean_reverbs, noise_reverbs
