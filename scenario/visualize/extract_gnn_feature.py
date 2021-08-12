import numpy as np
import torch
import util
import nnet
import tqdm
import os

def extract_gnn_feature(args):
    # get loader
    train_loader, dev_loader = util.get_visualize_loader(args)

    # create model
    model = nnet.get_model(args)
    nnet.print_summary(model)
    augment_model = nnet.Augmentation(args)
    
    # load model
    path = '/home/thanh/gnn-mvdr/ckpt/baseline/checkpoints/baseline.ckpt'
    checkpoint = torch.load(path)
    epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # reset output folder
    folder = '../gnn_feature'
    if os.path.exists(folder):
        os.system(f'rm -r {folder}')
    os.makedirs(folder)

    # generate z
    idx = 0
    with torch.no_grad():
        with tqdm.tqdm(dev_loader, unit="it") as pbar:
            pbar.set_description(f'Epoch {epoch}')
            for batch_idx, batch in enumerate(pbar):
                # extract data
                cleans, noises, rirs, rirs_idx = batch
                # augmenta data
                inputs, cleans, noises, clean_reverbs, noise_reverbs = augment_model(cleans, noises, rirs)
                # encode
                y_hat_device = model.encode(inputs)
                y_hat = y_hat_device.detach().cpu().numpy()
                # save data
                batch_size = rirs_idx.shape[0]
       
                for i in range(batch_size):
                    idx += 1
                    mdict = {
                        'feature': y_hat[i],
                        'label': rirs_idx[i],
                    }
                    path = os.path.join(folder, f'{idx}.npz')
                    np.savez(path, **mdict)

