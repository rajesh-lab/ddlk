############################################################
"""
DDLK code: Anonymous Github User
"""
############################################################
import logging
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .. import utils
from . import cde, mdn, swap


class DDLK(mdn.MDNModel):
    """Knockoff generator"""
    def __init__(self, hparams, q_joint):
        """
        hparams must contain:
        
            - n_components : list of dimension d, int
            - X_mu : d-dim tensor of means
            - X_sigma : d-dim tensor of std devs
            - n_params : list of parameters, or int
            - use_safe_softmax : Boolean
            - lr : list of learning rates, or float
            - gumbel_lr : float
            - weight_decay : list or float
            - scheduler_patience : list or int
            - scheduler_cooldown : list or int
            - reg_entropy : entropy regularization
            - q_joint : str or MDNJoint object
            - swapper : str, must be one of 'gumbel', 'random'
            - tau : float
            - hidden_layers : int
        """

        super().__init__()

        # assert essential hparams are present
        essentials = ['X_mu', 'X_sigma']
        for key in essentials:
            assert hasattr(hparams, key), f'hparams is missing {key}...'

        # insert default hparams if certain keys are not present
        defaults = {
            'n_components': 5,
            'n_params': 50,
            'use_safe_softmax': True,
            'lr': 1e-2,
            'gumbel_lr': 1e-2,
            'weight_decay': 1e-6,
            'scheduler_patience': 40,
            'scheduler_cooldown': 10,
            'tau': 0.1,
            'init_type': 'residual',
            'swapper': 'gumbel',
            'reg_entropy': 0.1
        }
        for key, value in defaults.items():
            if not hasattr(hparams, key):
                setattr(hparams, key, value)

        # preprocess and register hyperparameters
        attributes = ['n_components', 'n_params']
        self.preprocess_hparams(hparams, attributes)

        # save q_joint for training
        self.q_joint = q_joint
        ## freeze joint model
        self.q_joint.freeze()

        # parametrics
        ## initialize d CDE networks
        if hasattr(self.hparams, 'init_type'):
            init_type = self.hparams.init_type
        else:
            init_type = 'fixed'
        if hasattr(self.hparams, 'hidden_layers'):
            hidden_layers = self.hparams.hidden_layers
        else:
            hidden_layers = 3
        self.init_conditionals(kind='knockoff',
                               init_type=init_type,
                               hidden_layers=hidden_layers)
        ## initialize swapper
        if hasattr(self.hparams, 'swapper'):
            if self.hparams.swapper == 'gumbel':
                self.swapper = swap.GumbelSwapper(self.hparams.d,
                                                  tau=self.hparams.tau)
            elif self.hparams.swapper == 'random':
                self.swapper = swap.RandomSwapper(self.hparams.d)
            else:
                raise NotImplementedError(
                    f'Swapper of type [{self.hparams.swapper}] not found...')

        # initialize history
        self.hparams.history = dict()

    def forward(self, x):
        """Dummy method. Should not be used in training"""
        raise NotImplementedError('.forward() method should not be called')

    def get_conditionals(self):
        out = []
        for j in range(self.hparams.d):
            cde_j = getattr(self, f'cde_{j}')
            out.append(cde_j)
        return out

    def sample(self, x):
        x_cond = x

        for j in range(self.hparams.d):
            cde_j = getattr(self, f'cde_{j}')
            # sample x_j_tilde | x, x_1_tilde, ..., x_{j-1}_tilde
            x_j_tilde = cde_j.sample(x_cond)
            # condition on x_j_tilde as well
            x_cond = torch.cat([x_cond, x_j_tilde.reshape(-1, 1)], axis=1)

        # return only x_tilde
        return x_cond[:, self.hparams.d:]

    def log_prob(self, x, x_tilde, detach_params=False):
        lp = 0.
        for j in range(self.hparams.d):
            cde_j = getattr(self, f'cde_{j}')
            # compute log probabilitiy of x_j_tilde | x, x_1_tilde, ..., x_{j-1}_tilde
            x_cond = torch.cat([x, x_tilde[:, :j]], axis=1)
            lp += cde_j.log_prob(x_cond,
                                 x_tilde[:, j],
                                 detach_params=detach_params)

        return lp

    def training_step(self, batch, batch_idx, val_step=False):
        xb, = batch

        # KL terms
        ## log q_joint(x)
        lp_xb = self.q_joint.log_prob(xb)

        ## log q_knockoff(~x | x)
        ### sampled knockoff ~x | x
        xb_tilde = self.sample(xb)
        lp_xb_tilde = self.log_prob(xb, xb_tilde, detach_params=True)

        ## log q_joint(u)
        ### swap
        ub, ub_tilde = self.swapper(xb, xb_tilde)
        lp_ub = self.q_joint.log_prob(ub)

        ## log q_knockoff(~u | u)
        lp_ub_tilde = self.log_prob(ub, ub_tilde)

        # define loss. Add regularization only when training
        lmbda = 1.0 if val_step else (1.0 + self.hparams.reg_entropy)
        loss = (lp_xb + lmbda * lp_xb_tilde - lp_ub - lp_ub_tilde).mean()

        tensorboard_logs = {'loss': loss}
        return OrderedDict({'loss': loss, 'log': tensorboard_logs})

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx, val_step=True)

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        out = dict(val_loss=avg_loss)
        out['log'] = tensorboard_logs
        out['progress_bar'] = tensorboard_logs

        # save history
        for log_stat in ['val_loss']:
            if log_stat not in self.hparams.history:
                self.hparams.history[log_stat] = []
        ## save validation loss
        self.hparams.history['val_loss'].append(avg_loss.item())

        return out

    def on_after_backward(self):
        # change direction of gradient for swapping parameters
        for param in self.swapper.parameters():
            param.grad *= -1

    def on_epoch_end(self):
        # anneal tau parameter
        if isinstance(self.swapper, swap.GumbelSwapper):
            self.swapper.update_tau()

    def configure_optimizers(self):
        # if using gumbel swapper, use separate learning rate for gumbel parameters
        if isinstance(self.swapper, swap.GumbelSwapper):
            cde_parameters = []
            for cde_j in self.get_conditionals():
                cde_parameters += list(cde_j.parameters())
            gumbel_parameters = self.swapper.parameters()

            parameters = [{
                'params': cde_parameters
            }, {
                'params': gumbel_parameters,
                'lr': self.hparams.gumbel_lr
            }]
        else:
            parameters = self.parameters()

        optimizer = optim.Adam(parameters,
                               lr=self.hparams.lr,
                               betas=(0.9, 0.999),
                               eps=1e-08,
                               weight_decay=self.hparams.weight_decay,
                               amsgrad=False)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.6,
            patience=self.hparams.scheduler_patience,
            verbose=True,
            threshold=0.05,
            cooldown=self.hparams.scheduler_cooldown,
            min_lr=1e-9,
        )
        return [optimizer], [scheduler]


if __name__ == "__main__":
    pass
