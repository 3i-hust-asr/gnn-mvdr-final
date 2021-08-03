from joblib import Parallel, delayed
import numpy as np
import tqdm
import os

from .util import *

def gen_data(args):
    # step 1: get list of wav for training
    clean_list = get_list('librispeech_360', args)
    # noise_list = get_list('audioset', args)

    # step 2: generate clean (path, start_at) of train, dev, test set
    #         and noise
    config = {
        'clean': split_clean_data(clean_list, args),
        # 'noise': split_noise_data(noise_list, args),
    }

    # step 3a: clear the mixed folder
    # os.system('rm -rf ../mixed')

    # step 3b: cut and save the clean/noise audio to mixed folder
    for mode in tqdm.tqdm(['train', 'dev', 'test']):
        print(f'[+] mode={mode}, name=clean')
        folder = create_mixed_folder(mode, 'clean')
        def f(i, line):
            try:
                path, start = line
                x = clip_data(get_firstchannel_read(path), start, 6 * 16000)
                path = os.path.join(folder, f'{i}.npz')
                np.savez_compressed(path, x=x)
            except:
                pass
        Parallel(n_jobs=os.cpu_count())(
            delayed(f)(i, line) for i, line in enumerate(config['clean'][mode]))

        # print(f'[+] mode={mode}, name=noise')
        # folder = create_mixed_folder(mode, 'noise')
        # def f(i, line):
        #     x = get_firstchannel_read(line)
        #     path = os.path.join(folder, f'{i}.npz')
        #     np.savez_compressed(path, x=x)
        # Parallel(n_jobs=os.cpu_count())(
        #     delayed(f)(i, line) for i, line in enumerate(config['noise'][mode]))
