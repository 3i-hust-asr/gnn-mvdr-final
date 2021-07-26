from .augmentation import Augmentation
from .baseline import *

def get_model(args):
    if args.model == 'baseline':
        model = GNNFaS()
    else:
        raise NotImplementedError
    return model

def print_summary(model, verbose=False):
    if verbose:
        print(model)
    print('Trainable parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    print('Non-trainable parameters:', sum(p.numel() for p in model.parameters() if not p.requires_grad))
