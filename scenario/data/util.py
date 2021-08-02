import soundfile as sf
import numpy as np
import traceback
import librosa
import tqdm
import os

def get_data_folder(name, args):
    simu_dir = os.path.join(args.data_dir, 'simu')
    simu = os.listdir(simu_dir)
    simu_rirs_dir = os.path.join(args.data_dir, 'simu_rirs')
    simu_rirs = os.listdir(simu_rirs_dir)
    if name in simu:
        return os.path.join(simu_dir, name)
    elif name in simu_rirs:
        return os.path.join(simu_rirs_dir, name)

def save_final_list(items, name):
    folder = '/tmp/'
    path = os.path.join(folder, f'{name}.final_list')
    with open(path, 'w') as fp:
        fp.write('\n'.join(items))

def load_final_list(name):
    folder = '/tmp/'
    path = os.path.join(folder, f'{name}.final_list')
    if os.path.exists(path):
        return open(path).read().strip().split('\n')

def recursive_walk(rootdir):
    for r, dirs, files in os.walk(rootdir):
        for f in files:
            yield os.path.join(r, f)

def get_list(name, args):
    '''get list for config file
    then filter only files that exists'''
    final_list = load_final_list(name)
    if final_list is None:
        config_list = open(f'data/{name}.name').read().strip().split('\n')
        folder = get_data_folder(name, args)
        exist_list = [path for path in recursive_walk(folder) if (path.endswith('.wav') or path.endswith('.flac'))]
        final_list = []
        for path in tqdm.tqdm(exist_list):
            if os.path.basename(path) in config_list:
                final_list.append(path)
        save_final_list(final_list, name)
    return final_list

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

def get_start_time(line, segment_length=6):
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

def clip_data(data, start, segment_length):
    '''
    according the start point and segment_length to split the data
    args:
        data: numpy.array
        start: -2, -1, [0,...., L - 1]
        segment_length: int
    return:
        tgt: numpy.array
    '''
    tgt = np.zeros(segment_length)
    data_len = data.shape[0]
    if start == -2:
        """
        this means segment_length // 4 < data_len < segment_length // 2
        padding to A_A_A
        """
        if data_len < segment_length//3:
            data = np.pad(data, [0, segment_length//3 - data_len])
            tgt[:segment_length//3] += data
            st = segment_length//3
            tgt[st:st+data.shape[0]] += data
            st = segment_length//3 * 2
            tgt[st:st+data.shape[0]] += data
        
        else:
            """
            padding to A_A
            """
            # st = (segment_length//2 - data_len) % 101
            # tgt[st:st+data_len] += data
            # st = segment_length//2 + (segment_length - data_len) % 173
            # tgt[st:st+data_len] += data
            data = np.pad(data, [0, segment_length//2 - data_len])
            tgt[:segment_length//2] += data
            st = segment_length//2
            tgt[st:st+data.shape[0]] += data
    
    elif start == -1:
        '''
        this means segment_length < data_len*2
        padding to A_A
        '''
        if data_len % 4 == 0:
            tgt[:data_len] += data
            tgt[data_len:] += data[:segment_length-data_len]
        elif data_len % 4 == 1:
            tgt[:data_len] += data
        elif data_len % 4 == 2:
            tgt[-data_len:] += data
        elif data_len % 4 == 3:
            tgt[(segment_length-data_len)//2:(segment_length-data_len)//2+data_len] += data
    
    else:
        tgt += data[start:start+segment_length]
    
    return tgt

def split_clean_data(clean_list, args):
    # Training set
    train_lines = []
    done = False
    idx = 0
    for path in clean_list:
        for t in get_start_time(path):
            train_lines.append((path, t))
            if len(train_lines) >= args.num_train:
                done = True
                break
        idx += 1
        if done:
            break

    # Dev set
    dev_lines = []
    done = False
    start_idx = idx
    for path in clean_list[start_idx:]:
        for t in get_start_time(path):
            dev_lines.append((path, t))
            if len(dev_lines) >= args.num_dev:
                done = True
                break
        idx += 1
        if done:
            break

    # Test set
    test_lines = []
    done = False
    for path in clean_list[idx:]:
        for t in get_start_time(path):
            test_lines.append((path, t))
            if len(test_lines) >= args.num_test:
                done = True
                break
        if done:
            break
    config = {
        'train': train_lines,
        'dev'  : dev_lines,
        'test' : test_lines,
    }
    return config

def split_noise_data(noise_list, args):
    # Training set
    # 3 rooms
    end = args.num_train * 2
    train_lines = noise_list[:end]

    start = end
    end = start + args.num_dev * 2
    dev_lines   = noise_list[start: end]

    start = end
    end = start + args.num_test * 2
    test_lines = noise_list[start: end]

    config = {
        'train': train_lines,
        'dev'  : dev_lines,
        'test' : test_lines,
    }
    return config

def create_mixed_folder(name, mode):
    folder = f'../mixed/{mode}/{name}'
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder