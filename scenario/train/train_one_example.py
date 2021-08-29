from pprint import pprint
import torch
import tqdm

from .util import *
import util
import nnet

def train_one_example(args):
    print('train one example')
    # get data
    train_loader, dev_loader = util.get_loader(args)

    # create model
    model = nnet.get_model(args)
    nnet.print_summary(model)
    augment_model = nnet.Augmentation(args)

    # create optimizer
    optimizer = util.get_optimizer(model, args)

    # extract data
    for cleans, noises, rirs in train_loader:
        break

    # augment data
    inputs, cleans, noises, clean_reverbs, noise_reverbs = augment_model(cleans, noises, rirs)
    # convert to numpy
    x = inputs.detach().cpu().numpy()
    # y = cleans.detach().cpu().numpy()
    y = clean_reverbs[:, :, 0].detach().cpu().numpy()

    bar = tqdm.tqdm(range(args.num_epoch))
    
    with torch.autograd.set_detect_anomaly(True):
        for i in bar:
            # Train step
            optimizer.zero_grad()
            # compute loss
            # loss_dict = model.compute_loss(mix=inputs, clean=cleans, noise=noises)
            loss = model.compute_loss(mix=inputs, clean=clean_reverbs[:, :, 0])
            # backward and update weight
            loss.backward()
            # clip grad norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

            # Validation step
            # compute loss
            with torch.no_grad():
                y_hat_device = model(inputs)
                
            y_hat = y_hat_device.detach().cpu().numpy()
            # compute metrics
            metrics = compute_metrics(x, y, y_hat, args)
            # compute mean metrics
            for key in metrics:
                metrics[key] = metrics[key].mean()
                
            # if i % 10 == 0:
            bar.set_postfix(**metrics)
