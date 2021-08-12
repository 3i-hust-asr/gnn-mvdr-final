from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import os


def visualize_gnn_feature(args):
    # load data
    folder = '../gnn_feature'
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
    np.save('X_embedded_3d.npy', X_embedded)
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1])
    plt.show()
