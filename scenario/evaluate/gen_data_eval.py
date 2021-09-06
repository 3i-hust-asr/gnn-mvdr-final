from ..data_.gen_data import get_selected_list, get_wav_in_data_dir
from joblib import Parallel, delayed
import numpy as np
import time
import tqdm
import os

from ..data_.mix_wav_util import *
import util
import nnet

def get_wav_list(name, args, mode='dev'):
    print(f'[+] generating new wav list: {name} {mode}')
    selected_list = get_selected_list(name, args, mode=mode)
    wav_list = get_wav_in_data_dir(name, args)
    final_wav_list = []

    for path in tqdm.tqdm(wav_list):
        if os.path.basename(path) in selected_list:
            final_wav_list.append(path)
    return final_wav_list


def gen_data(args, mode='dev'):
    rirs = ['linear', 'circle', 'non_uniform']
    for rir in rirs:
        path = os.path.join(f'../config/{mode}/{rir}.list')
        if os.path.exists(path):
            continue
        rir_list = get_wav_list(rir, args, mode=mode)
        print('[+] getting rir wav list {} #{}={}'.format(mode, rir, len(rir_list)))
        with open(path, 'w') as fp:
            fp.write('\n'.join(rir_list))

    # for rir in rirs:
    #     mix_wav(rir, mode=mode)

    augment_model = nnet.Augmentation(args)

    for rir in rirs:
        loader = util.get_eval_loader(rir, args)
        folder = f'../mixed/{mode}/eval/{rir}'
        os.makedirs(folder, exist_ok=True)
        with tqdm.tqdm(loader, unit="it") as pbar:
            pbar.set_description(f'{mode} {rir}')
            for i, batch in enumerate(pbar):
                cleans, noises, rirs = batch
                inputs, cleans, noises, clean_reverbs, noise_reverbs = augment_model(cleans, noises, rirs)
                # convert to numpy
                x = inputs.detach().cpu().numpy()
                y = clean_reverbs.detach().cpu().numpy()
                # print(x.shape, y.shape)
                # exit()
                path = os.path.join(folder, f'data-{i}.npz')
                np.savez_compressed(path, x=x, y=y)



def mix_wav(rir, mode='dev'):
    ls = open(f'../config/{mode}/{rir}.list').read().strip().split('\n')
    folder = f'../mixed/{mode}/{rir}'
    os.makedirs(folder, exist_ok=True)
    print(f'mix_wav {mode} {rir}')
    def f(i, line):
        x = audioread(line)
        path = os.path.join(folder, f'rir-{i}.npz')
        np.savez_compressed(path, x=x)
    Parallel(n_jobs=os.cpu_count())(delayed(f)(i, line) for i, line in enumerate(ls))


def gen_data_eval(args):
    gen_data(args, 'dev')

