from joblib import Parallel, delayed
import numpy as np
import time
import os

from .mix_wav_util import *

def mix_wav_(args, mode='train'):
    folder = f'../mixed/{mode}'
    for _ in ['clean', 'rir', 'noise']:
        os.makedirs(f'{folder}/{_}', exist_ok=True)

    tic = time.time()
    ls = open(f'../config/{mode}/clean.list').read().strip().split('\n')
    def f(i, line):
        path, start = line.split(' ')
        start = int(start)
        segment_length = 6 * 16000
        x = get_firstchannel_read(path)
        x = clip_data(x, start, segment_length)
        path = os.path.join(folder, 'clean', f'clean-{i}.npz')
        np.savez_compressed(path, x=x)
        if i % 1000 == 0 and i > 0:
            elapsed = time.time() - tic
            remain = (len(ls) - i) / i * elapsed
            print('[+] {}/{} elapsed={} remain={}'.format(i, len(ls), elapsed, remain))

    Parallel(n_jobs=os.cpu_count())(delayed(f)(i, line) for i, line in enumerate(ls))

    ls = open(f'../config/{mode}/noise.list').read().strip().split('\n')
    def f(i, line):
        x = get_firstchannel_read(line)
        path = os.path.join(folder, 'noise', f'noise-{i}.npz')
        np.savez_compressed(path, x=x)
        if i % 1000 == 0 and i > 0:
            elapsed = time.time() - tic
            remain = (len(ls) - i) / i * elapsed
            print('[+] {}/{} elapsed={} remain={}'.format(i, len(ls), elapsed, remain))
    Parallel(n_jobs=os.cpu_count())(delayed(f)(i, line) for i, line in enumerate(ls))

    ls = open(f'../config/{mode}/rir.list').read().strip().split('\n')
    def f(i, line):
        x = audioread(line)
        path = os.path.join(folder, 'rir', f'rir-{i}.npz')
        np.savez_compressed(path, x=x)
        if i % 1000 == 0 and i > 0:
            elapsed = time.time() - tic
            remain = (len(ls) - i) / i * elapsed
            print('[+] {}/{} elapsed={} remain={}'.format(i, len(ls), elapsed, remain))
    Parallel(n_jobs=os.cpu_count())(delayed(f)(i, line) for i, line in enumerate(ls))

def mix_wav(args):
    for mode in ['dev', 'train']:
        mix_wav_(args, mode)
