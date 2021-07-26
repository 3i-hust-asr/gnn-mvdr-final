from joblib import Parallel, delayed
import rir_generator as rir
import numpy as np

from .util import *

def get_source_position(room_dimension):
    # distance from source position and wall is 0.5 meter
    return np.random.rand() * (np.array(room_dimension) - 1) + 0.5

def get_random_direction(dim=2):
    direction = np.random.rand(dim)
    direction = direction / np.sqrt(np.sum(direction ** 2))
    return direction

def get_mic_positions(room_dimension, num_mic=8, d_mic=0.04):
    origin = get_source_position(room_dimension)
    origin = np.array(origin)
    direction = get_random_direction() * d_mic
    # scale ra 8 mic
    mic_positions = [list(origin.copy())]
    current_mic_position = list(origin.copy())
    for mic in range(num_mic - 1):
        current_mic_position[:2] += direction
        mic_positions.append(list(current_mic_position.copy()))
    return mic_positions

def generate_rir_wrapper(mic_positions, source_position, room_dimension):
    return rir.generate(c=340,                          # Sound velocity (m/s)
                        fs=16000,                       # Sample rate (samples/s)
                        r=mic_positions,                # Microphone position(s) [x y z] (m)
                        s=source_position,              # Source position [x y z] (m)
                        L=room_dimension,               # Room dimensions [x y z] (m)
                        reverberation_time=0.5,         # Reverberation time (s)
                        nsample=8000)                   # Number of output samples

def generate_one_rir(room_dimension):
    clean_position = get_source_position(room_dimension)
    noise_position = get_source_position(room_dimension)
    mic_positions  = get_mic_positions(room_dimension)
    clean_rir      = generate_rir_wrapper(mic_positions, clean_position, room_dimension)
    noise_rir      = generate_rir_wrapper(mic_positions, noise_position, room_dimension)
    rir = np.concatenate([clean_rir, noise_rir], axis=1)
    return rir    

def f(mode, room_dimension, idx):
    folder = create_mixed_folder(mode, 'rir')
    x      = generate_one_rir(room_dimension)
    path   = os.path.join(folder, f'{idx}.npz')
    np.savez_compressed(path, x=x)

def gen_rir(args):
    # configuration
    room_dimensions = {
        'train': [(3, 3, 2), (5, 4, 6), (8, 9, 10)],
        'dev'  : [(5, 8, 3), (4, 7, 8)],
        'test' : [(4, 5, 3), (6, 8, 5)]
    }
    modes = ['train', 'dev', 'test']

    num_sample = {
        'train': 1200,
        'dev'  : 1500,
        'test' : 1500,
    }

    # generate rirs
    for mode in modes:
        args_list = []
        idx = 0
        for room_dimension in room_dimensions[mode]:
            for i in range(num_sample[mode]):
                args_list.append((mode, room_dimension, idx))
                idx += 1
        Parallel(n_jobs=os.cpu_count())(
            delayed(f)(*args) for args in args_list)
