import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
from .odst import ODST, GAM_ODST, GAMAttODST, GAMAtt3ODST
from . import nn_utils


class ODSTBlock(nn.Sequential):
    def __init__(self, input_dim, layer_dim, num_layers, num_classes=1, addi_tree_dim=0,
                 max_features=None, output_dropout=0.0, flatten_output=True,
                 last_as_output=False, init_bias=False, add_last_linear=False,
                 last_dropout=0., **kwargs):
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
        self.init_bias = init_bias
        self.add_last_linear = add_last_linear
        self.last_dropout = last_dropout

        if init_bias:
            val = torch.tensor(0.) if num_classes == 1 \
                else torch.full([num_classes], 0., dtype=torch.float32)
            self.bias = nn.Parameter(val, requires_grad=False)

        self.last_w = None
        if add_last_linear or addi_tree_dim < 0:
            # Happens when more outputs than intermediate tree dim
            self.last_w = nn.Parameter(torch.empty(
                num_layers * layer_dim * (num_classes + addi_tree_dim), num_classes))
            nn.init.xavier_uniform_(self.last_w)

        # Record which params need gradient
        self.named_params_requires_grad = set()
        for name, param in self.named_parameters():
            if param.requires_grad:
                self.named_params_requires_grad.add(name)

    def create_layers(self, input_dim, layer_dim, num_layers,
                      tree_dim, max_features=None, **kwargs):
        layers = []
        for i in range(num_layers):
            # Last layer only has num_classes dim
            oddt = ODST(input_dim, layer_dim, tree_dim=tree_dim, flatten_output=True,
                        **kwargs)
            input_dim = min(input_dim + layer_dim * tree_dim, max_features or float('inf'))
            layers.append(oddt)
        return layers

    def forward(self, x, return_outputs_penalty=False):
        outputs = self.run_with_layers(x)

        if not self.flatten_output:
            num_output_trees = self.layer_dim if self.last_as_output \
                else self.num_layers * self.layer_dim
            outputs = outputs.view(*outputs.shape[:-1], num_output_trees,
                                   self.num_classes + self.addi_tree_dim)
        # We can do weighted sum instead of just simple averaging
        if self.last_w is not None:
            last_w = self.last_w
            if self.training and self.last_dropout > 0.:
                last_w = F.dropout(last_w, self.last_dropout)
            result = torch.einsum(
                'bd,dc->bc',
                outputs.reshape(outputs.shape[0], -1),
                last_w
            ).squeeze_(-1)
        else:
            outputs = outputs[..., :self.num_classes]
            # ^--[batch_size, num_trees, num_classes]
            result = outputs.mean(dim=-2).squeeze_(-1)

        if self.init_bias:
            result += self.bias.data

        if return_outputs_penalty:
            # Average over batch, num_outputs_units
            output_penalty = (outputs ** 2).mean()
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

    def set_bias(self, y_train):
        ''' Set the bias term for GAM output as logodds of y. '''
        assert self.init_bias

        y_cls, counts = np.unique(y_train, return_counts=True)
        bias = np.log(counts / np.sum(counts))
        if len(bias) == 2:
            bias = bias[1] - bias[0]

        self.bias.data = torch.tensor(bias, dtype=torch.float32)

    def freeze_all_but_lastw(self):
        for name, param in self.named_parameters():
            if param.requires_grad and 'last_w' not in name:
                param.requires_grad = False

    def unfreeze(self):
        for name, param in self.named_parameters():
            if name in self.named_params_requires_grad:
                param.requires_grad = True

    # def get_response_l2_penalty(self):
    #     penalty = 0.
    #     for layer in self:
    #         penalty += layer.get_response_l2_penalty()
    #     return penalty

    def get_num_trees_assigned_to_each_feature(self):
        '''
        Return num of trees assigned to each feature in GAM.
        Return a vector of size equal to the in_features
        '''
        if isinstance(self, ODSTBlock):
            return None

        num_trees = [l.get_num_trees_assigned_to_each_feature() for l in self]
        return torch.stack(num_trees)

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
            addi_tree_dim=args.addi_tree_dim + getattr(args, 'data_addi_tree_dim', 0),
            depth=args.depth, flatten_output=False,
            choice_function=nn_utils.entmax15,
            init_bias=(getattr(args, 'init_bias', False)
                       and args.problem == 'classification'),
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
                           gen=lambda args: int(ch([4, 6]))),
            'addi_tree_dim': dict(short_name='td',
                             gen=lambda args: int(ch([0, 1, 2]))),
            'lr': dict(short_name='lr', gen=lambda args: 1e-3),
        }
        return rs_hparams

    @classmethod
    def add_model_specific_results(cls, results, args):
        results['depth'] = args.depth
        return results


class GAMBlock(ODSTBlock):
    def create_layers(self, input_dim, layer_dim, num_layers,
                      tree_dim, max_features=None, **kwargs):
        layers = []
        for i in range(num_layers):
            # Last layer only has num_classes dim
            oddt = GAM_ODST(input_dim, layer_dim, tree_dim=tree_dim,
                            flatten_output=True, **kwargs)
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
        assert args.arch in ['GAM', 'GAMAtt', 'GAMAtt2', 'GAMAtt3'], 'Wrong arch: ' + args.arch

        choice_fn = getattr(nn_utils, args.choice_fn)(
            max_temp=1., min_temp=args.min_temp, steps=args.anneal_steps)
        kwargs = dict(
            input_dim=args.in_features,
            layer_dim=args.num_trees,
            num_layers=args.num_layers,
            num_classes=args.num_classes,
            addi_tree_dim=args.addi_tree_dim + getattr(args, 'data_addi_tree_dim', 0),
            depth=args.depth, flatten_output=False,
            choice_function=choice_fn,
            bin_function=nn_utils.entmoid15,
            output_dropout=args.output_dropout,
            last_dropout=getattr(args, 'last_dropout', 0.),
            colsample_bytree=args.colsample_bytree,
            selectors_detach=args.selectors_detach,
            fs_normalize=args.fs_normalize,
            last_as_output=args.last_as_output,
            init_bias=(getattr(args, 'init_bias', False)
                       and args.problem == 'classification'),
            add_last_linear=getattr(args, 'add_last_linear', False),
            save_memory=getattr(args, 'save_memory', 0),
            ga2m=getattr(args, 'ga2m', 0),
        )

        if args.arch in ['GAMAtt', 'GAMAtt2', 'GAMAtt3'] and 'dim_att' in args:
            kwargs['dim_att'] = args.dim_att

        model = cls(**kwargs)
        if not ret_step_callback:
            return model

        return model, choice_fn.temp_step_callback

    def get_interactions(self):
        interactions = [l.get_interactions() for l in self]
        interactions = [tmp for tmp in interactions if tmp is not None]
        if len(interactions) == 0:
            return None

        results = torch.unique(torch.cat(interactions, dim=0), dim=0)
        return results.cpu().numpy().tolist()

    @classmethod
    def add_model_specific_args(cls, parser):
        parser.add_argument("--colsample_bytree", type=float, default=1.)
        parser.add_argument("--output_dropout", type=float, default=0.)
        parser.add_argument("--last_dropout", type=float, default=0.)
        parser.add_argument("--last_as_output", type=int, default=0)
        parser.add_argument("--min_temp", type=float, default=1e-2)
        parser.add_argument("--anneal_steps", type=int, default=4000)

        parser.add_argument("--choice_fn", default='EM15Temp',
                            help="Choose the dataset.",
                            choices=['GSMTemp', 'SMTemp', 'EM15Temp'])

        parser.add_argument("--selectors_detach", type=int, default=0)
        parser.add_argument("--fs_normalize", type=int, default=1)
        parser.add_argument("--init_bias", type=int, default=1)
        parser.add_argument("--add_last_linear", type=int, default=1)

        # Use GA2M
        parser.add_argument("--ga2m", type=int, default=0)

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
        def colsample_bytree_gen(args):
            if not args.ga2m:
                return ch([0.5, 0.1, 1e-5])

            choices = [1., 0.5, 0.1]
            new_choices = [c for c in choices if (args.in_features * c) > 1]
            return ch(new_choices)

        rs_hparams = {
            # 'arch': dict(short_name='', gen=lambda args: np.random.choice(['GAM', 'GAMAtt'])),
            'seed': dict(short_name='s', gen=lambda args: int(np.random.randint(100))),
            # 'seed': dict(short_name='s', gen=lambda args: 2),  # Fix seed; see other hparams
            'num_layers': dict(short_name='nl',
                               gen=lambda args: int(ch([2, 3, 4, 5]))),
            'num_trees': dict(short_name='nt',
                              # gen=lambda args: int(ch([4096, 8192, 16384, 32768, 32768*2]))),
                              gen=lambda args: int(ch([500, 1000, 2000, 4000])) // args.num_layers),
            'addi_tree_dim': dict(short_name='td',
                                  gen=lambda args: int(ch([0, 1, 2]))),
                                  # gen=lambda args: 0),
            'depth': dict(short_name='d', gen=lambda args: int(ch([2, 4, 6]))),
            'output_dropout': dict(short_name='od',
                                   gen=lambda args: ch([0., 0.1, 0.2])),
            'last_dropout': dict(short_name='ld',
                                 gen=lambda args: (0. if not args.add_last_linear
                                                   else ch([0., 0.1, 0.2, 0.3]))),
            'colsample_bytree': dict(short_name='cs', gen=colsample_bytree_gen),
            'lr': dict(short_name='lr', gen=lambda args: ch([0.01, 0.005])),
            # 'last_as_output': dict(short_name='lo', gen=lambda args: int(ch([0, 1]))),
            'last_as_output': dict(short_name='lo', gen=lambda args: 0),
            # 'anneal_steps': dict(short_name='an', gen=lambda args: int(ch([2000, 4000, 6000]))),
            'l2_lambda': dict(short_name='la',
                              gen=lambda args: float(ch([1e-5, 1e-6, 1e-7, 0.]))),
            'pretrain': dict(short_name='pt'),
            'pretraining_ratio': dict(
                short_name='pr',
                # gen=lambda args: float(ch([0.1, 0.15, 0.2])) if args.pretrain else 0),
                # gen=lambda args: 0.15 if args.pretrain else 0,
            ),
            'masks_noise': dict(
                short_name='mn',
                # gen=lambda args: float(ch([0., 0.1, 0.2])) if args.pretrain else 0),
                gen=lambda args: 0.1 if args.pretrain else 0),
            'opt_only_last_layer': dict(
                short_name='ol',
                # gen=lambda args: (int(ch([0, 1])) if args.pretrain else 0)),
                gen=lambda args: 0),
            'add_last_linear': dict(
                short_name='ll',
                gen=lambda args: (1 if (args.pretrain or args.arch == 'GAM')
                                  else int(ch([0, 1]))),
            ),
        }
        return rs_hparams

    @classmethod
    def add_model_specific_results(cls, results, args):
        results['anneal_steps'] = args.anneal_steps
        return results


class GAMAttBlock(GAMBlock):
    def create_layers(self, input_dim, layer_dim, num_layers,
                      tree_dim, max_features=None, **kwargs):
        layers = []
        prev_in_features = 0
        for i in range(num_layers):
            # Last layer only has num_classes dim
            oddt = GAMAttODST(input_dim, layer_dim, tree_dim=tree_dim,
                              flatten_output=True,
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
        ch = np.random.choice
        rs_hparams.update({
            'dim_att': dict(short_name='da',
                            gen=lambda args: int(ch([8, 16, 32]))),
            # 'add_last_linear': dict(
            #     short_name='ll',
            #     gen=lambda args: (1 if args.pretrain else int(ch([0, 1]))),
            # ),
            # 'add_last_linear': dict(short_name='ll', gen=lambda args: 1),
            # 'colsample_bytree': dict(short_name='cs',
            #                          gen=lambda args: ch([0.5, 0.1])),
        })
        return rs_hparams


class GAMAtt2Block(GAMAttBlock):
    pass


class GAMAtt3Block(GAMAttBlock):
    def create_layers(self, input_dim, layer_dim, num_layers,
                      tree_dim, max_features=None, **kwargs):
        layers = []
        prev_in_features = 0
        for i in range(num_layers):
            # Last layer only has num_classes dim
            oddt = GAMAtt3ODST(input_dim, layer_dim, tree_dim=tree_dim,
                               flatten_output=True,
                               prev_in_features=prev_in_features, **kwargs)
            layers.append(oddt)
            prev_in_features += layer_dim * tree_dim
        return layers
