import os
import time
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_latest_file, iterate_minibatches, check_numpy, process_in_chunks
from .nn_utils import to_one_hot
from collections import OrderedDict
from copy import deepcopy
from tensorboardX import SummaryWriter
from apex import amp
import json
from os.path import join as pjoin, exists as pexists
import argparse
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from sklearn.metrics import roc_auc_score, log_loss
from . import nn_utils, arch



class Trainer(nn.Module):
    def __init__(self, model, loss_function, experiment_name=None, warm_start=False,
                 Optimizer=torch.optim.Adam, optimizer_params={},
                 lr=0.01, lr_warmup_steps=-1, verbose=False,
                 n_last_checkpoints=1, step_callbacks=[], fp16=0,
                 l2_lambda=0., **kwargs):
        """
        :type model: torch.nn.Module
        :param loss_function: the metric to use in trainnig
        :param experiment_name: a path where all logs and checkpoints are saved
        :param warm_start: when set to True, loads last checpoint
        :param Optimizer: function(parameters) -> optimizer
        :param verbose: when set to True, produces logging information
        """
        super().__init__()
        self.model = model
        self.loss_function = loss_function
        self.verbose = verbose
        self.lr = lr
        self.lr_warmup_steps = lr_warmup_steps

        # When using fp16, there are some params if not filtered out by requires_grad
        # will produce error
        self.opt = Optimizer([p for p in self.model.parameters() if p.requires_grad],
                             lr=lr, **optimizer_params)
        self.step = 0
        self.n_last_checkpoints = n_last_checkpoints
        self.step_callbacks = step_callbacks
        self.l2_lambda = l2_lambda
        self.fp16 = fp16

        if experiment_name is None:
            experiment_name = 'untitled_{}.{:0>2d}.{:0>2d}_{:0>2d}:{:0>2d}'.format(*time.gmtime()[:5])
            if self.verbose:
                print('using automatic experiment name: ' + experiment_name)

        self.experiment_path = pjoin('logs/', experiment_name)
        # if not warm_start and experiment_name != 'debug':
        #     assert not os.path.exists(self.experiment_path), 'experiment {} already exists'.format(experiment_name)
        # self.writer = SummaryWriter(self.experiment_path, comment=experiment_name)
        if fp16:
            self.model, self.opt = amp.initialize(
                self.model, self.opt, opt_level='O1')
        if warm_start:
            self.load_checkpoint()

    def save_checkpoint(self, tag=None, path=None, mkdir=True, **kwargs):
        assert tag is None or path is None, "please provide either tag or path or nothing, not both"
        if tag is None and path is None:
            tag = "temp_{}".format(self.step)
        if path is None:
            path = pjoin(self.experiment_path, "checkpoint_{}.pth".format(tag))
        if mkdir:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(OrderedDict([
            ('model', self.model.state_dict(**kwargs)),
            ('opt', self.opt.state_dict()),
            ('step', self.step),
        ] + ([] if not self.fp16 else [('amp', amp.state_dict())])), path)
        if self.verbose:
            print("Saved " + path)
        return path

    def load_checkpoint(self, tag=None, path=None, **kwargs):
        assert tag is None or path is None, "please provide either tag or path or nothing, not both"
        if tag is None and path is None:
            path = get_latest_file(pjoin(self.experiment_path, 'checkpoint_temp_[0-9]*.pth'))
            if path is None:
                return self

        elif tag is not None and path is None:
            path = pjoin(self.experiment_path, "checkpoint_{}.pth".format(tag))

        checkpoint = torch.load(path)

        self.model.load_state_dict(checkpoint['model'], **kwargs)
        self.opt.load_state_dict(checkpoint['opt'])
        self.step = int(checkpoint['step'])
        if self.fp16 and 'amp' in checkpoint:
            amp.load_state_dict(checkpoint['amp'])

        if self.verbose:
            print('Loaded ' + path)
        return self

    def average_checkpoints(self, tags=None, paths=None, out_tag='avg', out_path=None):
        assert tags is None or paths is None, "please provide either tags or paths or nothing, not both"
        assert out_tag is not None or out_path is not None, "please provide either out_tag or out_path or both, not nothing"
        if tags is None and paths is None:
            paths = self.get_latest_checkpoints(
                pjoin(self.experiment_path, 'checkpoint_temp_[0-9]*.pth'), self.n_last_checkpoints)
        elif tags is not None and paths is None:
            paths = [pjoin(self.experiment_path, 'checkpoint_{}.pth'.format(tag)) for tag in tags]

        checkpoints = [torch.load(path) for path in paths]
        averaged_ckpt = deepcopy(checkpoints[0])
        for key in averaged_ckpt['model']:
            values = [ckpt['model'][key] for ckpt in checkpoints]
            averaged_ckpt['model'][key] = sum(values) / len(values)

        if out_path is None:
            out_path = pjoin(self.experiment_path, 'checkpoint_{}.pth'.format(out_tag))
        torch.save(averaged_ckpt, out_path)

    def get_latest_checkpoints(self, pattern, n_last=None):
        list_of_files = glob.glob(pattern)
        assert len(list_of_files) > 0, "No files found: " + pattern
        return sorted(list_of_files, key=os.path.getctime, reverse=True)[:n_last]

    def remove_old_temp_checkpoints(self, number_ckpts_to_keep=None):
        if number_ckpts_to_keep is None:
            number_ckpts_to_keep = self.n_last_checkpoints
        paths = self.get_latest_checkpoints(pjoin(self.experiment_path, 'checkpoint_temp_[0-9]*.pth'))
        paths_to_delete = paths[number_ckpts_to_keep:]

        for ckpt in paths_to_delete:
            os.remove(ckpt)

    def train_on_batch(self, *batch, device, update=True):
        # Tune the learning rate
        if self.lr_warmup_steps > 0 and self.step < self.lr_warmup_steps:
            cur_lr = self.lr * (self.step + 1) / self.lr_warmup_steps
            self.set_lr(cur_lr)

        x_batch, y_batch = batch
        x_batch = torch.as_tensor(x_batch, device=device)
        y_batch = torch.as_tensor(y_batch, device=device)

        self.model.train()

        # Read that it's faster...
        for group in self.opt.param_groups:
            for p in group['params']:
                p.grad = None
        # self.opt.zero_grad()

        if self.l2_lambda <= 0.:
            loss = self.loss_function(self.model(x_batch), y_batch).mean()
        else:
            logits, op = self.model(x_batch, return_output_penalty=True)
            loss = self.loss_function(logits, y_batch).mean()
            loss += self.l2_lambda * op

        if self.fp16:
            with amp.scale_loss(loss, self.opt) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        # If not updating, zeroing out all the gradient!
        # if not update:
        #     self.opt.zero_grad()

        if update:
            self.opt.step()
            self.step += 1
            # self.writer.add_scalar('train loss', loss.item(), self.step)
            for c in self.step_callbacks:
                c(self.step)
        
        return {'loss': loss.item()}

    def evaluate_classification_error(self, X_test, y_test, device, batch_size=4096):
        ''' This is for evaluation of binary error '''
        X_test = torch.as_tensor(X_test, device=device)
        y_test = check_numpy(y_test)
        self.model.train(False)
        with torch.no_grad():
            logits = process_in_chunks(self.model, X_test, batch_size=batch_size)
            logits = check_numpy(logits)
            error_rate = (y_test != (logits >= 0)).mean()
            # error_rate = (y_test != np.argmax(logits, axis=1)).mean()
        return error_rate

    def evaluate_negative_auc(self, X_test, y_test, device, batch_size=4096):
        X_test = torch.as_tensor(X_test, device=device)
        y_test = check_numpy(y_test)
        self.model.train(False)
        with torch.no_grad():
            logits = process_in_chunks(self.model, X_test, batch_size=batch_size)
            logits = check_numpy(logits)

            # assert logits.shape[1] == 2, 'logits shape is not binary! %d' % logits.shape[1]
            # logit_diff = logits[:, 1] - logits[:, 0]
            auc = roc_auc_score(y_test, logits)

        return -auc

    def evaluate_mse(self, X_test, y_test, device, batch_size=4096):
        X_test = torch.as_tensor(X_test, device=device)
        y_test = check_numpy(y_test)
        self.model.train(False)
        with torch.no_grad():
            prediction = process_in_chunks(self.model, X_test, batch_size=batch_size)
            prediction = check_numpy(prediction)
            error_rate = ((y_test - prediction) ** 2).mean()
        return error_rate

    def evaluate_logloss(self, X_test, y_test, device, batch_size=512):
        X_test = torch.as_tensor(X_test, device=device)
        y_test = check_numpy(y_test)
        self.model.train(False)
        with torch.no_grad():
            logits = F.softmax(process_in_chunks(self.model, X_test, batch_size=batch_size), dim=1)
            logits = check_numpy(logits)
            y_test = torch.tensor(y_test)
            logloss = log_loss(check_numpy(to_one_hot(y_test)), logits)
        return logloss

    def decrease_lr(self, ratio=0.1, min_lr=1e-6):
        if self.lr == min_lr:
            return

        self.lr *= ratio
        if self.lr < min_lr:
            self.lr = min_lr
        self.set_lr(self.lr)

    def set_lr(self, lr):
        for g in self.opt.param_groups:
            g['lr'] = lr

    @classmethod
    def load_best_model_from_trained_dir(cls, the_dir):
        ''' Follow the model architecture in main.py '''
        hparams = json.load(open(pjoin(the_dir, 'hparams.json')))

        model = getattr(arch, hparams['arch'] + 'Block').load_model_by_hparams(hparams)

        best_ckpt = pjoin(the_dir, 'checkpoint_{}.pth'.format('best'))
        if not pexists(best_ckpt):
            print('NO BEST CHECKPT EXISTS in {}!'.format(best_ckpt))
            return None

        tmp = torch.load(best_ckpt, map_location='cpu')
        model.load_state_dict(tmp['model'])
        return model
