from sklearn.metrics import silhouette_score
import numpy as np
import os

def load(mode):
    folder = f'../feature/{mode}'
    X = []
    Y = []
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        mat = np.load(path)
        X.append(mat['feature'].reshape(-1))
        Y.append(mat['label'])
    X = np.array(X)
    Y = np.array(Y)
    X_embedded = np.load(f'../embedded/{mode}_3d.npy')
    return X, X_embedded, Y

def silhouette(args):
    modes = [
        'train_z_in',
        'train_z_out',
        'pretrain_z_in',
        'pretrain_z_out',
        'train_2_z_in',
        'train_2_z_out',
        'noise_reverb_z_in',
        'noise_reverb_z_out',
        'clean_reverb_z_in',
        'clean_reverb_z_out',
    ]
    for mode in modes:
        X, X_embedded, Y = load(mode)
        score = silhouette_score(X_embedded, Y)
        print(mode, score)
