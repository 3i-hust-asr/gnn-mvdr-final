from .augmentation import Augmentation

def get_model(args):
    model = None
    return model

def print_summary(model):
    print('Trainable parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    print('Non-trainable parameters:', sum(p.numel() for p in model.parameters() if not p.requires_grad))
