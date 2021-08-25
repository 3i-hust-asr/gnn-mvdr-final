from joblib import Parallel, delayed
import numpy as np
import time
import os

from .mix_wav_util import *

def get_train_noise_list(noise, args, type='train'):
    train_noise = get_wav_list(noise, args, type=type)
    print('[+] getting noise wav list #{}={}'.format(noise, len(train_noise)))
    return train_noise

def get_train_rir_list(rir, args):
    train_rir = get_wav_list(rir, args)
    print('[+] getting rir wav list #{}={}'.format(rir, len(train_rir)))
    return train_rir

def mix_wav_(args):
    folder = '../mixed/'
    if not os.path.exists(folder):
        os.makedirs(folder)

    tic = time.time()
    ls = open('../config/train.list').read().strip().split('\n')
    def f(i, line):
        path, start = line.split(' ')
        start = int(start)
        segment_length = 6 * 16000
        x = get_firstchannel_read(path)
        x = clip_data(x, start, segment_length)
        path = os.path.join(folder, f'clean-{i}.npz')
        np.savez_compressed(path, x=x)
        if i % 1000 == 0 and i > 0:
            elapsed = time.time() - tic
            remain = (len(ls) - i) / i * elapsed
            print('[+] {}/{} elapsed={} remain={}'.format(i, len(ls), elapsed, remain))

    Parallel(n_jobs=os.cpu_count())(delayed(f)(i, line) for i, line in enumerate(ls))

    ls = open('../config/noise.list').read().strip().split('\n')
    def f(i, line):
        x = get_firstchannel_read(line)
        path = os.path.join(folder, f'noise-{i}.npz')
        np.savez_compressed(path, x=x)
        if i % 1000 == 0 and i > 0:
            elapsed = time.time() - tic
            remain = (len(ls) - i) / i * elapsed
            print('[+] {}/{} elapsed={} remain={}'.format(i, len(ls), elapsed, remain))
    Parallel(n_jobs=os.cpu_count())(delayed(f)(i, line) for i, line in enumerate(ls))

    ls = open('../config/rir.list').read().strip().split('\n')
    def f(i, line):
        x = audioread(line)
        path = os.path.join(folder, f'rir-{i}.npz')
        np.savez_compressed(path, x=x)
        if i % 1000 == 0 and i > 0:
            elapsed = time.time() - tic
            remain = (len(ls) - i) / i * elapsed
            print('[+] {}/{} elapsed={} remain={}'.format(i, len(ls), elapsed, remain))
    Parallel(n_jobs=os.cpu_count())(delayed(f)(i, line) for i, line in enumerate(ls))

def mix_wav(args):
    mix_wav_(args)
