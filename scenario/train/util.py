from joblib import Parallel, delayed
from pystoi.stoi import stoi
from pprint import pprint
from pesq import pesq
import numpy as np
import os

def get_ckpt_folder(args):
    folder = f'../ckpt/{args.model}/checkpoints'
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder

def get_ckpt_name(args):
    return '{}'.format(args.model)

def get_logs_folder(args):
    return os.path.join(get_ckpt_folder(args).replace('checkpoints', 'logs'), get_ckpt_name(args))

def compute_segmented_si_snr(y, y_hat):
    y_norm = y / np.linalg.norm(y, ord=2, axis=1, keepdims=True)
    y_hat = y_hat / np.linalg.norm(y_hat, ord=2, axis=1, keepdims=True)


    s_target = (y_norm * y_hat).sum(axis=1, keepdims=True) * y_norm
    e_noise = y_hat - s_target

    si_snr = 20 * np.log10(
        np.linalg.norm(s_target, ord=2, axis=1) / np.linalg.norm(e_noise, ord=2, axis=1)
    )
    return si_snr

def compute_pesq(y, y_hat, args):
    return np.array(Parallel(n_jobs=os.cpu_count())(delayed(pesq)(16000, y_, y_hat_, 'wb') for y_, y_hat_ in zip(y, y_hat)))

def compute_stoi(y, y_hat, args):
    return np.array(Parallel(n_jobs=os.cpu_count())(delayed(stoi)(y_, y_hat_, 16000, extended=False) for y_, y_hat_ in zip(y, y_hat)))

def compute_estoi(y, y_hat, args):
    return np.array(Parallel(n_jobs=os.cpu_count())(delayed(stoi)(y_, y_hat_, 16000, extended=True) for y_, y_hat_ in zip(y, y_hat)))

def compute_metrics(x, y, y_hat, args):
    # initialize metrics
    metrics = {}
    # SISNR
    metrics['si_snr:enhanced'] = compute_segmented_si_snr(y, y_hat)
    metrics['si_snr:noisy'] = compute_segmented_si_snr(y, x[..., 0])
    # PESQ
    metrics['pesq:enhanced'] = compute_pesq(y, y_hat, args)
    # metrics['pesq:noisy'] = compute_pesq(y, x[..., 0], args)
    # STOI
    metrics['stoi:enhanced'] = compute_stoi(y, y_hat, args)
    # metrics['stoi:noisy'] = compute_stoi(y, x[..., 0], args)
    # ESTOI
    metrics['estoi:enhanced'] = compute_estoi(y, y_hat, args)
    # metrics['estoi:noisy'] = compute_estoi(y, x[..., 0], args)
    return metrics