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
def get_selected_list(clean, args, type='train'):
    folder = f'data/selected_lists/{type}'
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

def get_wav_in_data_dir(clean, args, type='train'):
    if type == 'train':
        return _get_wav_in_data_dir(clean, args)
    elif type == 'dev':
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
    folder = '../tmp'
    path = os.path.join(folder, name)
    if os.path.exists(path):
        return open(path).read().strip().split('\n')

def save(items, name):
    folder = '../tmp'
    if not os.path.exists(folder):
        os.makedirs(folder)
    path = os.path.join(folder, name)
    with open(path, 'w') as fp:
        fp.write('\n'.join(items))

def get_wav_list(clean, args, type='train'):
    final_wav_list = load('{}_{}'.format(clean, type))
    if final_wav_list is not None:
        print('\n[+] train clean {} loaded'.format(clean))
        return final_wav_list
    else:
        print('\n[+] generating new wav list {}'.format(clean))
        # get train wav
        selected_list = get_selected_list(clean, args, type=type)
        wav_list = get_wav_in_data_dir(clean, args, type=type)

        final_wav_list = []
        for path in tqdm(wav_list):
            if os.path.basename(path) in selected_list:
                final_wav_list.append(path)
        save(final_wav_list, '{}_{}'.format(clean, type))
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

def save_train_config_full(items, args, name='train'):
    if not os.path.exists('../config'):
        os.mkdir('../config')
    path = '../config/{}.list'.format(name)
    with open(path, 'a') as fp:
        fp.write('\n'.join(items))
        fp.write('\n')
##############################################################

def get_train_clean_list(args, type='train'):
    if type == 'train':
        cleans = ['aishell_1', 'librispeech_360', 'aishell_3']
    elif type == 'dev':
        cleans = ['clean']

    # generating all clean list
    train_clean = []
    for clean in cleans:
        ls = get_wav_list(clean, args, type=type)
        train_clean += ls
        print('[+] getting clean wav list #{}={}, #total={} type={}'.format(clean, len(ls), len(train_clean), type))

    if not os.path.exists(f'../config/{type}.list'):
        def f(train_path):
            Fields = []
            for start_time in get_start_time(train_path, 6 * 16000):
                fields = [train_path, start_time]
                fields = [str(_) for _ in fields]
                Fields.append(fields)
            return [' '.join(fields) for fields in Fields]

        print(f'[+] generate config lines parallely type={type}')
        step = 12800
        for i in tqdm(range(0, len(train_clean), step)):
            tmp_paths = train_clean[i: i + step]
            #     
            Lines = Parallel(n_jobs=os.cpu_count())(delayed(f)(tp) for tp in tmp_paths)
            final_lines = []
            for lines in Lines:
                final_lines += lines
            # save data
            save_train_config_full(final_lines, args, name=f'{type}')

    lines = [line.split(' ') for line in open(f'../config/{type}.list').read().strip().split('\n')]
    print('\n[+] train list loaded, #examples={} type={}'.format(len(lines), type))
    return lines

def get_train_noise_list(noise, args, type='train'):
    train_noise = get_wav_list(noise, args, type=type)
    print('[+] getting noise wav list #{}={}'.format(noise, len(train_noise)))
    return train_noise

def get_train_rir_list(rir, args):
    train_rir = get_wav_list(rir, args)
    print('[+] getting rir wav list #{}={}'.format(rir, len(train_rir)))
    return train_rir

def gen_data_(args, type='train'):
    noises = ['musan_noise', 'musan_music', 'audioset']
    rirs = ['linear', 'circle', 'non_uniform']
    clean_start_list = get_train_clean_list(args)

    noise_list = []
    rir_list = []
    for noise in noises:
        for rir in rirs:
            noise_list += get_train_noise_list(noise, args)
            rir_list   += get_train_rir_list(rir, args)
    path = os.path.join('../config/noise.list')
    with open(path, 'w') as fp:
        fp.write('\n'.join(noise_list))
    path = os.path.join('../config/rir.list')
    with open(path, 'w') as fp:
        fp.write('\n'.join(rir_list))

def gen_data(args):
    # args.num_sample = 5000
    gen_data_(args, type='train')
    gen_data_(args, type='dev')
