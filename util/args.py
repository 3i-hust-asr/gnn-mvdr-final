import argparse
import torch
import os

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--scenario', type=str, default='mix_wav')

    parser.add_argument('--num_worker', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_sample', type=int, default=1)
    
    parser.add_argument('--model', type=str, default='mvdr')

    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--clip_grad_norm', type=float, default=5.0)
    parser.add_argument('--pretrain_path', type=str, required=False)
    parser.add_argument('--num_epoch', type=int, default=500)
    parser.add_argument('--limit_train_batch', type=int, default=-1)
    parser.add_argument('--limit_val_batch', type=int, default=-1)
    parser.add_argument('--log_iter', type=int, default=100)
    parser.add_argument('--non_ipd', action='store_true')

    parser.add_argument('--clear_cache', action='store_true')

    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    args.device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

    return args