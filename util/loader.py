from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import torch
import time
import os

class NoisyDataset(Dataset):

    def __init__(self, args, mode='train'):
        self.clean_path = [os.path.join(f'../mixed/clean/{mode}', f) for f in os.listdir(f'../mixed/clean/{mode}')]
        self.noise_path = [os.path.join(f'../mixed/noise/{mode}', f) for f in os.listdir(f'../mixed/noise/{mode}')]
        self.rir_path   = [os.path.join(f'../mixed/rir/{mode}', f) for f in os.listdir(f'../mixed/rir/{mode}')]
        # self.clean_path = [os.path.join(f'../mixed/clean/train', f) for f in os.listdir(f'../mixed/clean/train')]
        # self.noise_path = [os.path.join(f'../mixed/noise/train', f) for f in os.listdir(f'../mixed/noise/train')]
        # self.noise_path = list(sorted(self.noise_path))
        # self.rir_path   = [os.path.join(f'../mixed/rir/train', f) for f in os.listdir(f'../mixed/rir/train')]
        # self.rir_path   = list(sorted(self.rir_path))
        # self.rir_path   = self.rir_path[:args.limit_rir]
        self.args = args
        self.mode = mode

    def __len__(self):
        return len(self.clean_path)

    def __getitem__(self, idx):
        clean_idx = idx % len(self.clean_path)
        noise_idx = idx % len(self.noise_path)
        rir_idx = idx % len(self.rir_path)

        # clean
        clean = torch.tensor(np.load(self.clean_path[clean_idx])['x'], dtype=torch.float32)

        # noise + padding
        noise = torch.zeros_like(clean, dtype=torch.float32)
        noise_np = np.load(self.noise_path[noise_idx])['x']
        noise_np = noise_np[:len(clean)]
        noise[:len(noise_np)] = torch.tensor(noise_np, dtype=torch.float32)

        # rir
        rir   = torch.tensor(np.load(self.rir_path[rir_idx])['x'], dtype=torch.float32)
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
                               shuffle=args.shuffle,
                               collate_fn=collate_fn,
                               batch_size=args.batch_size,
                               num_workers=args.num_worker)
    
    dev_dataset = NoisyDataset(args, mode='dev')
    dev_loader  = DataLoader(dataset=dev_dataset, 
                             drop_last=True, 
                             shuffle=False,
                             collate_fn=collate_fn,
                             batch_size=args.batch_size,
                             num_workers=args.num_worker)

    return train_loader, dev_loader