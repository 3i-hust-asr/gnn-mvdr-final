from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import torch
import time
import os

class NoisyDataset(Dataset):

    def __init__(self, args, mode='train'):
        self.path = './mixed'
        self.clean_path = [f for f in os.listdir(self.path) if f.startswith('clean')]
        self.noise_path = [f for f in os.listdir(self.path) if f.startswith('noise')]
        self.rir_path   = [f for f in os.listdir(self.path) if f.startswith('rir')]

        self.args = args
        self.device = args.device
        self.mode = mode

    def __len__(self):
        if self.mode == 'train':
            return self.args.num_sample
        elif self.mode == 'dev':
            return self.args.num_sample // 50
        else:
            raise NotImplementedError

    def __getitem__(self, idx):
        clean_idx = np.random.randint(low=0, high=len(self.clean_path))
        noise_idx = np.random.randint(low=0, high=len(self.noise_path))
        rir_idx = np.random.randint(low=0, high=len(self.rir_path))

        # clean
        clean = torch.tensor(np.load(os.path.join(self.path, self.clean_path[clean_idx]))['x'], dtype=torch.float32)

        # noise + padding
        noise = torch.zeros_like(clean, dtype=torch.float32)
        noise_np = np.load(os.path.join(self.path, self.noise_path[noise_idx]))['x']
        noise_np = noise_np[:len(clean)]
        noise[:len(noise_np)] = torch.tensor(noise_np, dtype=torch.float32)

        # rir
        rir   = torch.tensor(np.load(os.path.join(self.path, self.rir_path[rir_idx]))['x'], dtype=torch.float32)
        return clean, noise, rir

def get_loader(args):

    def collate_fn(batch):
        clean = torch.stack([item[0] for item in batch])
        noise = torch.stack([item[1] for item in batch])
        rir = [item[2] for item in batch]
        return clean, noise, rir

    train_dataset = NoisyDataset(args, mode='train')
    train_loader  = DataLoader(dataset=train_dataset, 
                            drop_last=True, 
                            collate_fn=collate_fn,
                            batch_size=args.batch_size,
                            num_workers=args.num_worker)
    
    dev_dataset = NoisyDataset(args, mode='dev')
    dev_loader  = DataLoader(dataset=dev_dataset, 
                            drop_last=True, 
                            collate_fn=collate_fn,
                            batch_size=args.batch_size,
                            num_workers=args.num_worker)

    return train_loader, dev_loader