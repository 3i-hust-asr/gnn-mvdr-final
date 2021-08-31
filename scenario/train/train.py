from torch.utils.tensorboard.writer import SummaryWriter
import numpy as np
import torch
import tqdm
import os

from .util import *
import util
import nnet

torch.backends.cudnn.benchmark = True

class Trainer:

    def __init__(self, args):
        # save args
        self.args = args
        # clear cache
        self.clear_cache()
        # get loader
        self.train_loader, self.dev_loader = util.get_loader(args)
        # get model
        self.model = nnet.get_model(args)
        self.augment_model = nnet.Augmentation(args)
        nnet.print_summary(self.model)
        # get optimizer
        self.optimizer = util.get_optimizer(self.model, args)
        # get writer
        self.writer = SummaryWriter(get_logs_folder(args))
        # get iteration
        self.iteration = 0

    def train_step(self, batch, batch_idx):
        # extract data
        cleans, noises, rirs = batch

        # augmenta data
        inputs, cleans, noises, clean_reverbs, noise_reverbs = self.augment_model(cleans, noises, rirs)

        # 
        self.optimizer.zero_grad()
        # compute loss
        loss_dict = {'loss': self.model.compute_loss(mix=inputs, clean=clean_reverbs[:, :, 0])}
        # DEBUG
        if torch.isnan(loss_dict['loss']):
            print('ra nan roi', loss_dict['loss'])
            raise KeyboardInterrupt            
        # END DEBUG
        # backward and update weight
        loss_dict['loss'].backward()
        # clip grad norm
        self.clip_grad_norm()
        self.optimizer.step()

        return loss_dict

    def validation_step(self, batch, batch_idx, mode='dev'):
        with torch.no_grad():
            # extract data
            cleans, noises, rirs = batch
            # augmenta data
            inputs, cleans, noises, clean_reverbs, noise_reverbs = self.augment_model(cleans, noises, rirs)
            # convert to numpy
            x = inputs.detach().cpu().numpy()
            y = clean_reverbs[:, :, 0].detach().cpu().numpy()
            # compute loss
            y_hat_device = self.model(inputs)
            y_hat = y_hat_device.detach().cpu().numpy()
            # compute metrics
            metrics = compute_metrics(x, y, y_hat, self.args)
        cleaned_metrics = {}
        for key in metrics:
            cleaned_metrics[f'{mode}:{key}'] = metrics[key]
        return cleaned_metrics

    def limit_train_batch_hook(self, batch_idx):
        if self.args.limit_train_batch > 0:
            if batch_idx > self.args.limit_train_batch:
                return True
        return False

    def limit_val_batch_hook(self, batch_idx):
        if self.args.limit_val_batch > 0:
            if batch_idx > self.args.limit_val_batch:
                return True
        return False

    def get_checkpoint_path(self):
        ckpt_folder = get_ckpt_folder(self.args)
        ckpt_name = get_ckpt_name(self.args)
        return os.path.join(ckpt_folder, ckpt_name) + '.ckpt'

    def load_checkpoint(self):
        self.epoch = 0
        path = self.get_checkpoint_path()
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.epoch = checkpoint['epoch']
            self.iteration = checkpoint['iteration']
        if self.args.pretrain_path is not None:
            if os.path.exists(self.args.pretrain_path):
                checkpoint = torch.load(self.args.pretrain_path)
                self.model.load_state_dict(checkpoint['model_state_dict'])

    def clear_cache(self):
        ckpt_folder = get_ckpt_folder(self.args)
        logs_folder = get_logs_folder(self.args)
        if self.args.clear_cache:
            os.system(f'rm -rf {ckpt_folder} {logs_folder}')

    def clip_grad_norm(self):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)

    def write_dev_metric_to_tensorboard(self, epoch, metrics):
        # compute average
        for key in metrics:
            metrics[key] = np.mean(metrics[key])
        # display
        print('Evaluate epoch:{}: si_snr={:0.2f} pesq={:0.2f},  stoi={:0.2f}, estoi={:0.2f}' \
            .format(epoch, metrics['dev:si_snr:enhanced'], metrics['dev:pesq:enhanced'],
                           metrics['dev:stoi:enhanced'], metrics['dev:estoi:enhanced']))
        # write to tensorboard
        self.writer.add_scalars('validation metric', metrics, epoch)

    def write_train_metric_to_tensorboard(self, loss_dicts):
        for key in loss_dicts:
            loss_dicts[key] = np.mean(loss_dicts[key])
        self.writer.add_scalars('training metric', loss_dicts, self.iteration)

    def _fit(self):
        # load checkpoint
        self.load_checkpoint()

        for epoch in range(self.epoch, self.args.num_epoch + 1):
            
            ##########################################################################################
            # evalute
            if self.args.evaluate and (epoch % self.args.eval_iter == 0):
                self.model.eval()
                all_metrics = {}
                with tqdm.tqdm(self.dev_loader, unit="it") as pbar:
                    pbar.set_description(f'Evaluate epoch - dev {epoch}')
                    metrics = {}
                    for batch_idx, batch in enumerate(pbar):
                        # validate
                        batch_metrics = self.validation_step(batch, batch_idx, mode='dev')
                        # accumulate valilation metrics
                        for key in batch_metrics:
                            if key not in metrics.keys():
                                metrics[key] = []
                        for key in batch_metrics:
                            metrics[key] += batch_metrics[key].tolist()
                        pbar.set_postfix(si_snr=np.mean(metrics['dev:si_snr:enhanced']))
                        # limit train batch hook
                        if self.limit_val_batch_hook(batch_idx):
                            break
                all_metrics.update(metrics)

                with tqdm.tqdm(self.train_loader, unit="it") as pbar:
                    pbar.set_description(f'Evaluate epoch - train {epoch}')
                    for batch_idx, batch in enumerate(pbar):
                        # validate
                        batch_metrics = self.validation_step(batch, batch_idx, mode='train')
                        # accumulate valilation metrics
                        for key in batch_metrics:
                            if key not in metrics.keys():
                                metrics[key] = []
                        for key in batch_metrics:
                            metrics[key] += batch_metrics[key].tolist()
                        pbar.set_postfix(si_snr=np.mean(metrics['train:si_snr:enhanced']))
                        # limit train batch hook
                        if self.limit_val_batch_hook(batch_idx):
                            break
                all_metrics.update(metrics)

                # print epoch summary
                self.write_dev_metric_to_tensorboard(epoch, metrics)
                
            ##########################################################################################
            # train
            loss_dicts = None
            self.model.train()
            with tqdm.tqdm(self.train_loader, unit="it") as pbar:
                pbar.set_description(f'Epoch {epoch}')
                for batch_idx, batch in enumerate(pbar):

                    # perform training step
                    loss_dict = self.train_step(batch, batch_idx)
                    if loss_dicts is None:
                        loss_dicts = {}
                        for key in loss_dict:
                            loss_dicts[key] = []
                    for key in loss_dict:
                        loss_dicts[key].append(float(loss_dict[key].detach().cpu()))

                    # limit train batch hook
                    if self.limit_train_batch_hook(batch_idx):
                        break

                    # set postfix
                    kwargs = {}
                    for key in loss_dict:
                        kwargs[key] = float(loss_dict[key].detach().cpu())
                    pbar.set_postfix(**kwargs)

                    # log
                    self.epoch = epoch
                    self.iteration += 1
                    if self.iteration % self.args.log_iter == 0:
                        self.write_train_metric_to_tensorboard(loss_dicts)
                        loss_dicts = None
            # save checkpoint
            self.save_checkpoint()


            # set use cache to true
            ##########################################################################################

    def fit(self):
        try:
            self._fit()
        except KeyboardInterrupt: 
            pass
        self.save_checkpoint()

    def save_checkpoint(self):
        # save checkpoint
        torch.save({
            'iteration': self.iteration,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, self.get_checkpoint_path())
        # save checkpoint for each epoch
        torch.save({
            'iteration': self.iteration,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, self.get_checkpoint_path().replace('.ckpt', f'_epoch_{self.epoch}.ckpt'))
        print('[+] checkpoint saved')

def train(args):
    trainer = Trainer(args)
    trainer.fit()
