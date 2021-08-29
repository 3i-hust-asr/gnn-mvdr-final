from .augmentation import Augmentation
from .baseline import *
from .chanvl import *

def get_model(args):
    if args.model == 'baseline':
        model = GNNFaS(args)
    elif args.model == 'baseline1':
        model = GNNFaS1(args)
    elif args.model == 'tencent':
        model = Tencent()
    elif args.model == 'mvdr':
        model = Mvdr(args)
    else:
        raise NotImplementedError
    return model.to(args.device)

def print_summary(model, verbose=False):
    if verbose:
        print(model)
    print('Trainable parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    print('Non-trainable parameters:', sum(p.numel() for p in model.parameters() if not p.requires_grad))

