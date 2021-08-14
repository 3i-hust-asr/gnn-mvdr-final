from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import os


def visualize_gnn_feature(args):
    # load data
    # mode = 'noise_reverb_2_z_in'
    # mode = 'noise_reverb_2_z_out'
    # mode = 'noise_reverb_z_in'
    # mode = 'noise_reverb_z_out'
    # mode = 'clean_reverb_z_out'
    # mode = 'clean_reverb_z_in'
    # mode = 'train_2_z_out'
    mode = 'train_2_z_in'
    
    folder = f'../feature/{mode}/'
    X = []
    Y = []
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        mat = np.load(path)
        X.append(mat['feature'].reshape(-1))
        Y.append(mat['label'])
    X = np.array(X)
    Y = np.array(Y)
    X_embedded = TSNE(n_components=3).fit_transform(X)
    np.save(f'../embedded/{mode}_3d.npy', X_embedded)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(X_embedded[:, 0], X_embedded[:, 1], X_embedded[:, 2], c=Y)
    plt.title(mode)
    plt.show()
