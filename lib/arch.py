import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
from .odst import ODST, GAM_ODST, GAMAttODST
from . import nn_utils


class ODSTBlock(nn.Sequential):
    def __init__(self, input_dim, layer_dim, num_layers, num_classes=1, addi_tree_dim=0,
                 max_features=None, output_dropout=0.0, flatten_output=True,
                 last_as_output=False, **kwargs):
        layers = self.create_layers(input_dim, layer_dim, num_layers,
                                    tree_dim=num_classes + addi_tree_dim,
                                    max_features=max_features,
                                    **kwargs)
        super().__init__(*layers)
        self.num_layers, self.layer_dim, self.num_classes, self.addi_tree_dim = \
            num_layers, layer_dim, num_classes, addi_tree_dim
        self.max_features, self.flatten_output = max_features, flatten_output
        self.output_dropout = output_dropout
        self.last_as_output = last_as_output

    def create_layers(self, input_dim, layer_dim, num_layers,
                      tree_dim=1, max_features=None, **kwargs):
        layers = []
        for i in range(num_layers):
            oddt = ODST(input_dim, layer_dim, tree_dim=tree_dim, flatten_output=True,
                        **kwargs)
            input_dim = min(input_dim + layer_dim * tree_dim, max_features or float('inf'))
            layers.append(oddt)
        return layers

    def forward(self, x, return_outputs_penalty=False):
        outputs = self.run_with_layers(x)

        if not self.flatten_output:
            outputs = outputs.view(*outputs.shape[:-1], self.num_layers * self.layer_dim,
                                   self.num_classes + self.addi_tree_dim)
        outputs = outputs[..., :self.num_classes]
        result = outputs.mean(dim=-2).squeeze_(-1)

        if return_outputs_penalty:
            # Average over number of output_units
            output_penalty = (outputs ** 2).sum() / outputs.shape[-2]
            return result, output_penalty
        return result

    def run_with_layers(self, x):
        initial_features = x.shape[-1]
        for layer in self:
            layer_inp = x
            if self.max_features is not None:
                tail_features = min(self.max_features, layer_inp.shape[-1]) - initial_features
                if tail_features != 0:
                    layer_inp = torch.cat([layer_inp[..., :initial_features], layer_inp[..., -tail_features:]], dim=-1)
            h = layer(layer_inp)
            if self.training and self.output_dropout:
                h = F.dropout(h, self.output_dropout)
            x = torch.cat([x, h], dim=-1)

        outputs = h if self.last_as_output else x[..., initial_features:]
        return outputs

    def get_response_l2_penalty(self):
        penalty = 0.
        for layer in self:
            penalty += layer.get_response_l2_penalty()
        return penalty

    @classmethod
    def load_model_by_hparams(cls, args, ret_step_callback=False):
        if isinstance(args, dict):
            args = argparse.Namespace(**args)
        assert args.arch == 'ODST', 'Wrong arch: ' + args.arch

        model = ODSTBlock(
            input_dim=args.in_features,
            layer_dim=args.num_trees,
            num_layers=args.num_layers,
            num_classes=args.num_classes,
            addi_tree_dim=args.addi_tree_dim,
            depth=args.depth, flatten_output=False,
            choice_function=nn_utils.entmax15,
            # output_dropout=args.output_dropout,
            # colsample_bytree=args.colsample_bytree,
            bin_function=nn_utils.entmoid15,
        )

        if not ret_step_callback:
            return model

        return model, None

    @classmethod
    def add_model_specific_args(cls, parser):
        for action in parser._actions:
            if action.dest == 'lr':
                action.default = 1e-3
            # if action.dest == 'batch_size':
            #     action.default = 1024
        return parser

    @classmethod
    def get_model_specific_rs_hparams(cls):
        ch = np.random.choice

        rs_hparams = {
            'seed': dict(short_name='s', gen=lambda args: int(np.random.randint(100))),
            'num_layers': dict(short_name='nl',
                               gen=lambda args: int(ch([2, 4, 8]))),
            'num_trees': dict(
                short_name='nt',
                # gen=lambda args: int(ch([4096, 8192, 16384, 32768, 32768*2]))),
                gen=lambda args: int(ch([1024, 2048]) // args.num_layers)),
            'depth': dict(short_name='d',
                           gen=lambda args: int(ch([6, 8]))),
            'addi_tree_dim': dict(short_name='td',
                             gen=lambda args: int(ch([0, 1]))),
            'lr': dict(short_name='lr', gen=lambda args: 1e-3),
        }
        return rs_hparams

    @classmethod
    def add_model_specific_results(cls, results, args):
        results['depth'] = args.depth
        return results


class GAMBlock(ODSTBlock):
    def create_layers(self, input_dim, layer_dim, num_layers,
                      tree_dim=1, max_features=None, **kwargs):
        layers = []
        for i in range(num_layers):
            oddt = GAM_ODST(input_dim, layer_dim, tree_dim=tree_dim, flatten_output=True,
                            **kwargs)
            layers.append(oddt)
        return layers

    def run_with_layers(self, x):
        initial_features = x.shape[-1]
        prev_feature_selectors = None
        for layer in self:
            layer_inp = x
            h, feature_selectors = layer(
                layer_inp, prev_feature_selectors=prev_feature_selectors,
                return_feature_selectors=True)
            if self.training and self.output_dropout:
                h = F.dropout(h, self.output_dropout)
            x = torch.cat([x, h], dim=-1)

            prev_feature_selectors = feature_selectors \
                if prev_feature_selectors is None \
                else torch.cat([prev_feature_selectors, feature_selectors], dim=1)

        outputs = h if self.last_as_output else x[..., initial_features:]
        return outputs

    @classmethod
    def load_model_by_hparams(cls, args, ret_step_callback=False):
        if isinstance(args, dict):
            args = argparse.Namespace(**args)
        assert args.arch in ['GAM', 'GAMAtt'], 'Wrong arch: ' + args.arch

        choice_fn = getattr(nn_utils, args.choice_fn)(
            max_temp=1., min_temp=args.min_temp, steps=args.anneal_steps)
        kwargs = dict(
            input_dim=args.in_features,
            layer_dim=args.num_trees,
            num_layers=args.num_layers,
            num_classes=args.num_classes,
            addi_tree_dim=args.addi_tree_dim,
            depth=args.depth, flatten_output=False,
            choice_function=choice_fn,
            bin_function=nn_utils.entmoid15,
            output_dropout=args.output_dropout,
            colsample_bytree=args.colsample_bytree,
            selectors_detach=args.selectors_detach,
            fs_normalize=args.fs_normalize,
            last_as_output=args.last_as_output,
        )

        model_arch = GAMBlock if args.arch == 'GAM' else GAMAttBlock
        if args.arch == 'GAMAtt' and 'dim_att' in args:
            kwargs['dim_att'] = args.dim_att

        model = model_arch(**kwargs)
        if not ret_step_callback:
            return model

        return model, choice_fn.temp_step_callback

    @classmethod
    def add_model_specific_args(cls, parser):
        parser.add_argument("--colsample_bytree", type=float, default=1.)
        parser.add_argument("--output_dropout", type=float, default=0.)
        parser.add_argument("--last_as_output", type=int, default=0)
        parser.add_argument("--min_temp", type=float, default=1e-6)
        parser.add_argument("--anneal_steps", type=int, default=4000)

        parser.add_argument("--choice_fn", default='EM15Temp',
                            help="Choose the dataset.",
                            choices=['GSMTemp', 'SMTemp', 'EM15Temp'])

        parser.add_argument("--selectors_detach", type=int, default=0)
        parser.add_argument("--fs_normalize", type=int, default=1)

        # Change default value
        for action in parser._actions:
            if action.dest == 'lr':
                action.default = 0.01
            elif action.dest == 'lr_warmup_steps':
                action.default = 500
            elif action.dest == 'lr_decay_steps':
                action.default = 5000
            elif action.dest == 'early_stopping_rounds':
                action.default = 11000

        return parser

    @classmethod
    def get_model_specific_rs_hparams(cls):
        ch = np.random.choice
        rs_hparams = {
            # 'arch': dict(short_name='', gen=lambda args: np.random.choice(['GAM', 'GAMAtt'])),
            'seed': dict(short_name='s', gen=lambda args: int(np.random.randint(100))),
            # 'seed': dict(short_name='s', gen=lambda args: 2),  # Fix seed; see other hparams
            'num_layers': dict(short_name='nl',
                               gen=lambda args: int(ch([2, 3, 4]))),
            'num_trees': dict(short_name='nt',
                              # gen=lambda args: int(ch([4096, 8192, 16384, 32768, 32768*2]))),
                              gen=lambda args: int(ch([1000, 2000, 4000, 8000])) // args.num_layers),
            'addi_tree_dim': dict(short_name='td',
                                  gen=lambda args: int(ch([0, 1]))),
                                  # gen=lambda args: 0),
            'depth': dict(short_name='d', gen=lambda args: int(ch([2, 3, 4, 5]))),
            'output_dropout': dict(short_name='od',
                                   gen=lambda args: ch([0., 0.1, 0.2])),
            'colsample_bytree': dict(short_name='cs',
                                     gen=lambda args: ch([1., 0.5, 0.1, 0.00001])),
            'lr': dict(short_name='lr', gen=lambda args: ch([0.02, 0.01, 0.005])),
            'last_as_output': dict(short_name='lo', gen=lambda args: int(ch([0, 1]))),
            # 'anneal_steps': dict(short_name='an', gen=lambda args: int(ch([2000, 4000, 6000]))),
            # 'l2_lambda': dict(short_name='lda',
            #                   # gen=lambda args: float(ch([1e-9, 1e-10, 0.]))),
            #                   gen=lambda args: 0.),
        }
        return rs_hparams

    @classmethod
    def add_model_specific_results(cls, results, args):
        results['anneal_steps'] = args.anneal_steps
        return results


class GAMAttBlock(GAMBlock):
    def create_layers(self, input_dim, layer_dim, num_layers,
                      tree_dim=1, max_features=None, **kwargs):
        layers = []
        prev_in_features = 0
        for i in range(num_layers):
            oddt = GAMAttODST(input_dim, layer_dim, tree_dim=tree_dim, flatten_output=True,
                              prev_in_features=prev_in_features, **kwargs)
            layers.append(oddt)
            prev_in_features += layer_dim * tree_dim
        return layers

    @classmethod
    def add_model_specific_args(cls, parser):
        parser = super().add_model_specific_args(parser)
        parser.add_argument("--dim_att", type=int, default=64)
        return parser

    @classmethod
    def get_model_specific_rs_hparams(cls):
        rs_hparams = super().get_model_specific_rs_hparams()
        rs_hparams.update({
            'dim_att': dict(short_name='da',
                            gen=lambda args: int(np.random.choice([16, 64, 128]))),
        })
        return rs_hparams

