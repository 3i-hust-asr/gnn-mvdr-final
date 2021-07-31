from .augmentation import Augmentation
from .baseline import *

def get_model(args):
    if args.model == 'baseline':
        model = GNNFaS(args)
    elif args.model == 'baseline1':
        model = GNNFaS1(args)
    elif args.model == 'tencent':
        model = Tencent()
    else:
        raise NotImplementedError
    return model.to(args.device)

def print_summary(model, verbose=False):
    if verbose:
        print(model)
    print('Trainable parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    print('Non-trainable parameters:', sum(p.numel() for p in model.parameters() if not p.requires_grad))

# 00%|█| 500/500 [08:00<00:00,  1.04it/s, estoi:enhanced=0.653, pesq:enhanced=1.85, si_snr:enhanced=11.6, si_snr:noisy=5.94, stoi:enhanc