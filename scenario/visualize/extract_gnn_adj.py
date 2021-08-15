import numpy as np
import torch
import util
import nnet
import tqdm
import os

def extract_gnn_adj(args):
    # get loader
    train_loader, dev_loader = util.get_visualize_loader(args)

    # create model
    model = nnet.get_model(args)
    nnet.print_summary(model)
    augment_model = nnet.Augmentation(args)
    
    # load model
    path = 'ckpt/baseline.ckpt'
    checkpoint = torch.load(path)
    epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # reset output folder
    gcn_1_adj_folder  = '../feature/gcn_1_adj'
    gcn_2_adj_folder = '../feature/gcn_2_adj'
    for folder in [gcn_1_adj_folder, gcn_2_adj_folder]:
        os.system(f'rm -rf {folder}')
        os.makedirs(folder)

    # generate z
    idx = 0
    with torch.no_grad():
        with tqdm.tqdm(train_loader, unit="it") as pbar:
            pbar.set_description(f'Epoch {epoch}')
            for batch_idx, batch in enumerate(pbar):
                # extract data
                cleans, noises, rirs, rirs_idx = batch
                # augmenta data
                inputs, cleans, noises, clean_reverbs, noise_reverbs = augment_model(cleans, noises, rirs)
                # encode
                gcn_1_adj, gcn_2_adj  = model.forward_adj(inputs)
                gcn_1_adj = gcn_1_adj.detach().cpu().numpy()
                gcn_2_adj = gcn_2_adj.detach().cpu().numpy()
                # save data
                batch_size  = rirs_idx.shape[0]
                for i in range(batch_size):
                    idx += 1
                    
                    mdict = {
                        'feature': gcn_1_adj[i],
                        'label': rirs_idx[i],
                    }
                    path = os.path.join(gcn_1_adj_folder, f'{idx}.npz')
                    np.savez(path, **mdict)

                    mdict = {
                        'feature': gcn_2_adj[i],
                        'label': rirs_idx[i],
                    }
                    path = os.path.join(gcn_2_adj_folder, f'{idx}.npz')
                    np.savez(path, **mdict)
