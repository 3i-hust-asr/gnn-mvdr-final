import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_gnn_feature_v2(args):
    # load data
    modes = [
        'train_z_in',
        'train_z_out',
        'pretrain_z_in',
        'pretrain_z_out',
        'train_2_z_in',
        'train_2_z_out',
        'clean_reverb_z_in',
        'clean_reverb_z_out',
        'noise_reverb_z_in',
        'noise_reverb_z_out',
    ]
    mode = modes[0]

    folder = f'../feature-v2/feature/{mode}'
    X = []
    Y = []
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        mat = np.load(path)
        X.append(mat['feature'].reshape(-1))
        Y.append(mat['label'])
    X = np.array(X)
    Y = np.array(Y)

    # X_embedded = np.load('X_embedded.npy')

    # plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=Y)
    # plt.show()

    X_embedded = np.load(f'../feature-v2/embedded/{mode}_3d.npy')

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(X_embedded[:, 0], X_embedded[:, 1], X_embedded[:, 2], c=Y)
    plt.title(mode)
    plt.show()
