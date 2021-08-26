from joblib import Parallel, delayed
# from pprint import pprint as print
import soundfile as sf
from tqdm import tqdm
import numpy as np
import itertools
import traceback
import librosa
import random
import time
import os
#############################################################################
# CLEAN
#############################################################################
def get_selected_list(clean, args, mode='train'):
    folder = f'data/selected_lists/{mode}'
    path = os.path.join(folder, '{}.name'.format(clean))
    ls = open(path).read().strip().split('\n')
    return ls

def recursive_walk(rootdir):
    for r, dirs, files in os.walk(rootdir):
        for f in files:
            yield os.path.join(r, f)

def _get_wav_in_data_dir(clean, args):
    simu_dir = os.path.join(args.data_dir, 'simu')
    simu = os.listdir(simu_dir)
    simu_rirs_dir = os.path.join(args.data_dir, 'simu_rirs')
    simu_rirs = os.listdir(simu_rirs_dir)

    if clean in simu:
        folder = os.path.join(simu_dir, clean)
    elif clean in simu_rirs:
        folder = os.path.join(simu_rirs_dir, clean)
    else:
        pass
    return [path for path in recursive_walk(folder) if (path.endswith('.wav') or path.endswith('.flac'))]

def get_wav_in_data_dir(clean, args, mode='train'):
    if mode == 'train':
        return _get_wav_in_data_dir(clean, args)
    elif mode == 'dev':
        items = []
        if clean == 'clean':
            for clean in ['aishell_1', 'librispeech_360', 'aishell_3']:
                items += _get_wav_in_data_dir(clean, args)
        elif clean == 'noise':
            for clean in ['musan_noise', 'musan_music', 'audioset']:
                items += _get_wav_in_data_dir(clean, args)
        else:
            items += _get_wav_in_data_dir(clean, args)
    else:
        raise NotImplementedError
    return items

def load(name):
    path = os.path.join('../tmp', name)
    if os.path.exists(path):
        return open(path).read().strip().split('\n')
    return None

def save(items, name):
    folder = '../tmp'
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, name)
    with open(path, 'w') as fp:
        fp.write('\n'.join(items))

def get_wav_list(clean, args, mode='train'):
    final_wav_list = load('{}_{}'.format(clean, mode))
    if final_wav_list is not None:
        print('\n[+] {} {} loaded'.format(mode, clean))
        return final_wav_list
    else:
        print('\n[+] generating new wav list: {} {}'.format(mode, clean))
        # get train wav
        selected_list = get_selected_list(clean, args, mode=mode)
        wav_list = get_wav_in_data_dir(clean, args, mode=mode)

        final_wav_list = []
        for path in tqdm(wav_list):
            if os.path.basename(path) in selected_list:
                final_wav_list.append(path)
        save(final_wav_list, '{}_{}'.format(clean, mode))
    return final_wav_list

##############################################################
def get_firstchannel_read(path, fs=16000):
    '''
    args
        path: wav path
        fs: sample rate
    return
        wave_data: L
    '''
    wave_data, sr = sf.read(path)
    if sr != fs:
        if len(wave_data.shape) != 1:
            wave_data = wave_data.transpose((1, 0))
        wave_data = librosa.resample(wave_data, sr, fs)
        if len(wave_data.shape) != 1:
            wave_data = wave_data.transpose((1, 0))
    if len(wave_data.shape) > 1:
        wave_data = wave_data[:, 0]
    return wave_data

def get_start_time(line, segment_length):
    results = []
    try:
        path = line.strip()
        data = get_firstchannel_read(path)
        length = data.shape[0]

        if length < segment_length:
            if length * 2 < segment_length and length * 4 > segment_length:
                results.append(-2)
            elif length * 2 > segment_length:
                results.append(-1)
        else:
            sample_index = 0
            while sample_index + segment_length <= length:
                results.append(sample_index)
                sample_index += segment_length
            if sample_index < length:
                results.append(length - segment_length)
    except :
        traceback.print_exc()
    return results

def get_snr(snr_range=[0, 8]):
    return np.random.uniform(*snr_range)

def get_scale(scale_range=[0.2, 0.9]):
    return np.random.uniform(*scale_range)

def save_config_full(items, args, mode='train'):
    path = f'../config/{mode}/clean.list'
    with open(path, 'a') as fp:
        fp.write('\n'.join(items))
        fp.write('\n')
##############################################################

def get_clean_list(args, mode='train'):
    if mode == 'train':
        cleans = ['aishell_1', 'librispeech_360', 'aishell_3']
    elif mode == 'dev':
        cleans = ['clean']

    # generating all clean list
    clean_list = []
    for clean in cleans:
        ls = get_wav_list(clean, args, mode=mode)
        clean_list += ls
        print('[+] getting clean wav list #{}={}, #total={} mode={}'.format(clean, len(ls), len(clean_list), mode))

    if not os.path.exists(f'../config/{mode}.list'):
        def f(path):
            Fields = []
            for start_time in get_start_time(path, 6 * 16000):
                fields = [path, start_time]
                fields = [str(_) for _ in fields]
                Fields.append(fields)
            return [' '.join(fields) for fields in Fields]

        print(f'[+] generate config lines parallely mode={mode}')
        step = 12800
        for i in tqdm(range(0, len(clean_list), step)):
            tmp_paths = clean_list[i: i + step]
            #     
            Lines = Parallel(n_jobs=os.cpu_count())(delayed(f)(tp) for tp in tmp_paths)
            final_lines = []
            for lines in Lines:
                final_lines += lines
            # save data
            save_config_full(final_lines, args, mode)

def get_noise_list(args, mode='train'):
    if mode == 'train':
        noises = ['musan_noise', 'musan_music', 'audioset']
    elif mode == 'dev':
        noises = ['noise']
    noise_list = []
    for noise in noises:
        noise_list += get_wav_list(noise, args, mode=mode)
        print('[+] getting noise wav list {} #{}={}'.format(mode, noise, len(noise_list)))

    path = os.path.join(f'../config/{mode}/noise.list')
    with open(path, 'w') as fp:
        fp.write('\n'.join(noise_list))

def get_rir_list(args, mode='train'):
    rirs = ['linear', 'circle', 'non_uniform']
    rir_list = []
    for rir in rirs:
        rir_list += get_wav_list(rir, args, mode=mode)
    print('[+] getting rir wav list {} #{}={}'.format(mode, rir, len(rir_list)))

    path = os.path.join(f'../config/{mode}/rir.list')
    with open(path, 'w') as fp:
        fp.write('\n'.join(rir_list))

def gen_data_(args, mode='train'):
    get_clean_list(args, mode)
    get_rir_list(args, mode)
    get_noise_list(args, mode)

def gen_data(args):
    for mode in ['dev', 'train']:
        os.makedirs(f'../config/{mode}', exist_ok=True)
        gen_data_(args, mode)
