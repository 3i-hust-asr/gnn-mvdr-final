from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import os


def visualize_gnn_adj(args):
    # load data
    mode = 'gcn_2_adj'
    dimension = 2
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    
    os.makedirs('../embedded', exist_ok=True)
    path_X_embedded = f'../embedded/{mode}_{dimension}d.npy'

    if not os.path.exists(path_X_embedded):
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
        X_embedded = TSNE(n_components=dimension).fit_transform(X)
        np.save(f'../embedded/{mode}_{dimension}d.npy', X_embedded)
    else:
        folder = f'../feature/{mode}/'
        Y = []
        for name in os.listdir(folder):
            path = os.path.join(folder, name)
            mat = np.load(path)
            Y.append(mat['label'])
        Y = np.array(Y)

        X_embedded = np.load(path_X_embedded)

    # print(X_embedded.shape)
    X_embedded = X_embedded[:10000]
    Y = Y[:10000]
    fig = plt.figure()
    if dimension == 2:
        ax = fig.add_subplot()
        # ax.scatter(X_embedded[:, 0], X_embedded[:, 1], c=Y, label=np.unique(Y))
        for g in np.unique(Y):
            ix = np.where(Y == g)
            ax.scatter(X_embedded[ix, 0], X_embedded[ix, 1], c=colors[g], label=g)

    elif dimension == 3:
        ax = fig.add_subplot(projection=f'3d')
        # ax.scatter(X_embedded[:, 0], X_embedded[:, 1], X_embedded[:, 2], c=Y, label=np.unique(Y))
        for g in np.unique(Y):
            ix = np.where(Y == g)
            ax.scatter(X_embedded[ix, 0], X_embedded[ix, 1], X_embedded[ix, 2], c=colors[g], label=g)

    ax.legend()
    plt.title(mode)
    plt.show()
