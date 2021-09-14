import numpy as np
import torch
import time
import tqdm
import json
import os

from ..train.util import compute_metrics
import nnet
import util

def _evaluate(model_name, epoch, args):

    args.model = model_name
    model = nnet.get_model(args)
    nnet.print_summary(model)
    augment_model = nnet.Augmentation(args)

    ckpt_path = f'ckpt/{model_name}/checkpoints/{model_name}_epoch_{epoch}.ckpt'
    ckpt_path = f'ckpt/{model_name}_epoch_{epoch}.ckpt'
    if not os.path.exists(ckpt_path):
        raise ckpt_path
    checkpoint = torch.load(ckpt_path, map_location=args.device)
    print('Evaluate checkpoint:', ckpt_path)
    print(checkpoint['model_state_dict'].__dict__)
    for w in  checkpoint['model_state_dict']:
        print(w)
    # exit()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    all_metrics = {}

    for rir in ['linear', 'circle', 'non_uniform']:
        loader = util.get_mixed_loader(rir, args)
        # _, loader = util.get_loader(args)

        metrics = {}
        with tqdm.tqdm(loader, unit="it") as pbar:
            pbar.set_description(f'Evaluate {rir}')
            for i, batch in enumerate(pbar):
                with torch.no_grad():
                    # cleans, noises, rirs = batch
                    # inputs, _, _, clean_reverbs, _ = augment_model(cleans, noises, rirs)
                    # x = inputs.detach().cpu().numpy()
                    # y = clean_reverbs[:, :, 0].detach().cpu().numpy()
                    # y_hat_device = model(inputs)
                    # y_hat = y_hat_device.detach().cpu().numpy()

                    noisy, clean_reverb = batch
                    x = noisy.detach().cpu().numpy()
                    y = clean_reverb[:, :, 0].detach().cpu().numpy()
                    y_hat_device = model(noisy.to(args.device))
                    y_hat = y_hat_device.detach().cpu().numpy()

                    batch_metrics = compute_metrics(x, y, y_hat, args)

                for key in batch_metrics:
                    if key not in metrics.keys():
                        metrics[key] = []
                for key in batch_metrics:
                    metrics[key] += batch_metrics[key].tolist()
                pbar.set_postfix(si_snr=np.mean(metrics['si_snr:enhanced']))

        for key in metrics:
            metrics[key] = np.mean(metrics[key])
        all_metrics[rir] = metrics
        print(rir, metrics)
    return all_metrics


def evaluate(args):
    all_metrics = {
        'baseline': {},
        'tencent': {},
        'mvdr': {},
    }

    for epoch in [13]:
        metric = _evaluate('tencent', epoch, args)
        all_metrics['tencent'][f'epoch_{epoch}'] = metric

    for epoch in [6]:
        metric = _evaluate('baseline', epoch, args)
        all_metrics['baseline'][f'epoch_{epoch}'] = metric

    for epoch in [2]:
        metric = _evaluate('mvdr', epoch, args)
        all_metrics['mvdr'][f'epoch_{epoch}'] = metric

    with open('ckpt/result.json', 'w') as f:
        json.dump(all_metrics, f, indent=4)