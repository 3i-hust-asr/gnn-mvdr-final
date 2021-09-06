from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import torch
import time
import os

class NoisyDataset(Dataset):

    def __init__(self, args, mode='train'):
        self.args = args
        self.mode = mode
        self.clean_path = [os.path.join(f'../mixed/{mode}/clean', f) for f in os.listdir(f'../mixed/{mode}/clean')]
        self.noise_path = [os.path.join(f'../mixed/{mode}/noise', f) for f in os.listdir(f'../mixed/{mode}/noise')]
        self.rir_path   = [os.path.join(f'../mixed/{mode}/rir', f) for f in os.listdir(f'../mixed/{mode}/rir')]

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
        rir = torch.tensor(np.load(self.rir_path[rir_idx])['x'], dtype=torch.float32)
        return clean, noise, rir


class EvalDataset(Dataset):

    def __init__(self, rir, args, mode='dev'):
        self.args = args
        self.mode = mode
        self.clean_path = [os.path.join(f'../mixed/{mode}/clean', f) for f in os.listdir(f'../mixed/{mode}/clean')]
        self.noise_path = [os.path.join(f'../mixed/{mode}/noise', f) for f in os.listdir(f'../mixed/{mode}/noise')]
        self.rir_path = [os.path.join(f'../mixed/{mode}/{rir}', f) for f in os.listdir(f'../mixed/{mode}/{rir}')]
        # print(rir, mode, len(self.clean_path), self.clean_path[0])
        # print(rir, mode, len(self.noise_path), self.noise_path[0])
        # print(rir, mode, len(self.rir_path), self.rir_path[0])

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
        rir = torch.tensor(np.load(self.rir_path[rir_idx])['x'], dtype=torch.float32)
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


def get_eval_loader(rir, args):
    def collate_fn(batch):
        clean = torch.stack([item[0] for item in batch])
        noise = torch.stack([item[1] for item in batch])
        rir = [item[2] for item in batch]
        return clean, noise, rir
    return DataLoader(dataset=EvalDataset(rir, args), collate_fn=collate_fn, batch_size=1)