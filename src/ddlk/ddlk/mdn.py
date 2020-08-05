import argparse
from collections import OrderedDict
from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.core.lightning import LightningModule

from .. import utils
from . import cde, gmm


class LightMDN(nn.Module):
    """
    Barebones mixture density network
    """
    def __init__(self,
                 in_features=1,
                 n_components=5,
                 init_mu_sep=3,
                 init_mu=None):
        super().__init__()
        self.pi_transform = nn.Linear(in_features, n_components)
        self.mu_transform = nn.Linear(in_features, n_components)
        self.sigma_transform = nn.Linear(in_features, n_components)

        # initialize mdn with well-separated clusters
        ## equal probability across components
        self.pi_transform.weight.data.fill_(0.0)
        self.pi_transform.bias.data.fill_(1 / n_components)
        ## well separated mu
        if init_mu is None:
            init_mu = (1 - n_components) * init_mu_sep / 2 + torch.tensor(
                [(k - 1) * init_mu_sep for k in range(1, 1 + n_components)])
        else:
            if isinstance(init_mu, torch.Tensor):
                init_mu = init_mu
            else:
                init_mu = torch.tensor(init_mu)
        self.mu_transform.weight.data.fill_(0.0)
        self.mu_transform.bias.data = init_mu
        ## sigma = 1
        self.sigma_transform.weight.data.fill_(0.0)
        self.sigma_transform.bias.data.fill_(
            torch.log(torch.exp(torch.tensor(1.)) - 1.))

    def forward(self, x):
        pi = torch.softmax(self.pi_transform(x).relu(), axis=-1)
        mu = self.mu_transform(x)
        sigma = nn.functional.softplus(self.sigma_transform(x).relu())

        return pi, mu, sigma


class MDNModel(LightningModule):
    """Abstract class for MDN models"""
    def __init__(self):
        super().__init__()

    def my_device(self):
        return next(self.parameters()).device

    def preprocess_hparams(self, hparams, attributes):
        # preprocess and save hyperparameters
        self.hparams = hparams
        ## get dimension of data d
        self.hparams.d = self.hparams.X_mu.shape[-1]
        ## make hyperparameters lists of size d
        for attribute in attributes:
            if hasattr(self.hparams, attribute):
                value = getattr(self.hparams, attribute)
                if not isinstance(value, Iterable):
                    setattr(self.hparams, attribute, [value] * self.hparams.d)

    def init_conditionals(self,
                          kind='joint',
                          init_type='default',
                          hidden_layers=2):
        """
        input_mu and input_sigma must be of dimension = n_inputs
        """
        assert kind in {'joint',
                        'knockoff'}, f'kind must be one of {{joint, knockoff}}'
        if init_type == 'residual':
            assert kind == 'knockoff', f'residual initialization not defined for kind = {kind}'

        # create conditionals
        for j in range(self.hparams.d):
            # Define standardization constants
            ## y is the output of the CDE
            y_mu = self.hparams.X_mu[j]
            y_sigma = self.hparams.X_sigma[j]

            # X is the input of the CDE
            ## if the MDN is a joint distribution
            if kind == 'joint':
                if j == 0:
                    X_mu = torch.tensor([1.])
                    X_sigma = torch.tensor([1.])
                else:
                    X_mu = self.hparams.X_mu[:j]
                    X_sigma = self.hparams.X_sigma[:j]

                # set input dimension of CDE
                input_dim = max(1, j)
            ## if the MDN is a conditional distribution
            elif kind == 'knockoff':
                # concatenate X_mu and parts of X_mu again (for knockoffs)
                X_mu = torch.cat([self.hparams.X_mu, self.hparams.X_mu[:j]])
                X_sigma = torch.cat(
                    [self.hparams.X_sigma, self.hparams.X_sigma[:j]])

                # set input dimension of CDE
                input_dim = self.hparams.d + j

            hparams_j = argparse.Namespace(
                d=input_dim,
                n_components=self.hparams.n_components[j],
                X_mu=X_mu,
                X_sigma=X_sigma,
                y_mu=y_mu,
                y_sigma=y_sigma,
                n_params=self.hparams.n_params[j],
                use_safe_softmax=self.hparams.use_safe_softmax,
                init_type=init_type,
                hidden_layers=hidden_layers,
                j=j)

            cde_j = cde.CDE(hparams_j)

            setattr(self, f'cde_{j}', cde_j)


class MDNJoint(MDNModel):
    """Joint density"""
    def __init__(self, hparams):
        """
        hparams must contain:
            - n_components : list of dimension d, int
            - X_mu : d-dim tensor of means
            - X_sigma : d-dim tensor of std devs
            - n_params : list of parameters, or int
            - use_safe_softmax : Boolean
            - lr : list of learning rates, or float
            - weight_decay : list or float
            - scheduler_patience : list or int
            - scheduler_cooldown : list or int
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
            'lr': [5e-4] + [5e-4] * (hparams.X_mu.shape[-1] - 1),
            'weight_decay': 1e-6,
            'scheduler_patience': 10,
            'scheduler_cooldown': 2,
            'reg_lambda': 1e-7,
            'init_type': 'default',
            'dropout_rate': 0.0
        }
        for key, value in defaults.items():
            if not hasattr(hparams, key):
                setattr(hparams, key, value)

        # preprocess and register hyperparameters
        attributes = [
            'n_components', 'n_params', 'lr', 'weight_decay',
            'scheduler_patience', 'scheduler_cooldown', 'reg_lambda'
        ]
        self.preprocess_hparams(hparams, attributes)

        # initialize d CDE networks
        if hasattr(self.hparams, 'init_type'):
            init_type = self.hparams.init_type
        else:
            init_type = 'default'
        self.init_conditionals(kind='joint', init_type=init_type)

        # initialize history
        self.hparams.history = {j: dict() for j in range(self.hparams.d)}

    def forward(self, x):
        d = self.hparams.d

        out = []
        for j in range(d):
            cde_j = getattr(self, f'cde_{j}')

            if j == 0:
                inp_x = torch.ones(x.shape[0], 1)
            else:
                inp_x = x[:, :j]

            inp_x = cde_j.standardize(inp_x)
            out.append(cde_j(inp_x))

        return out

    def log_prob(self, x, detach_params=False):
        d = self.hparams.d

        lp = 0.
        # put input on gpu
        x_j_input = torch.ones(x.shape[0], 1)
        x_j_input = x_j_input.type_as(x)

        for j in range(d):
            x_j_output = x[:, j]
            cde_j = getattr(self, f'cde_{j}')

            lp += cde_j.log_prob(x_j_input,
                                 x_j_output,
                                 detach_params=detach_params)
            x_j_input = x[:, :j + 1]

        return lp

    def sample(self, sample_shape=torch.Size([1, 1])):
        x_m_j = torch.ones(sample_shape)
        for j in range(self.hparams.d):
            cde_j = getattr(self, f'cde_{j}')
            x_j = cde_j.sample(x_m_j)

            if j == 0:
                x_m_j = x_j.reshape(-1, 1)
            else:
                x_m_j = torch.cat([x_m_j, x_j.reshape(-1, 1)], axis=1)

        return x_m_j

    def training_step(self, batch, batch_idx, optimizer_idx):
        xb, = batch
        j = optimizer_idx

        # loss logging
        tensorboard_logs = dict()

        loss_j = 0
        cde_j = getattr(self, f'cde_{j}')

        # get inputs
        if j == 0:
            x_j_input = torch.ones(xb.shape[0], 1)
            x_j_input = x_j_input.type_as(xb)
        else:
            x_j_input = xb[:, :j]
        x_j_output = xb[:, j]

        # standardize inputs according to cde_j
        x_j_input, x_j_output = cde_j.standardize(x_j_input, x_j_output)

        # compute loss for the jth CDE
        pi, mu, sigma = cde_j(x_j_input)
        q_x_j = gmm.GaussianMixture(pi, mu, sigma)
        lp = q_x_j.log_prob(x_j_output) + torch.log(
            1 / self.hparams.X_sigma[j])
        loss_j += -lp.mean(axis=0)

        # regularization
        if self.hparams.reg_lambda[j] > 0:
            if self.hparams.n_components[j] > 1:
                loss_j += self.hparams.reg_lambda[j] * (
                    (pi / (1 - pi)).log()**2).mean()
                loss_j += self.hparams.reg_lambda[j] * (mu**2).mean()
                loss_j += self.hparams.reg_lambda[j] * (1 / sigma**2).mean()

        # record loss_j
        tensorboard_logs[f'loss_{j}'] = loss_j

        return OrderedDict({'loss': loss_j, 'log': tensorboard_logs})

    def validation_step(self, batch, batch_idx):
        xb, = batch

        # loss logging
        tensorboard_logs = dict()
        out = dict()

        loss = 0
        for j in range(self.hparams.d):
            loss_j = 0
            cde_j = getattr(self, f'cde_{j}')

            # get inputs
            if j == 0:
                x_j_input = torch.ones(xb.shape[0], 1)
                x_j_input = x_j_input.type_as(xb)
            else:
                x_j_input = xb[:, :j]
            x_j_output = xb[:, j]

            # get log prob
            lp = cde_j.log_prob(x_j_input, x_j_output)
            loss_j = -lp.mean(axis=0)

            # record loss_j
            tensorboard_logs[f'loss_{j}'] = loss_j
            out[f'loss_{j}'] = loss_j

            # add loss
            loss += loss_j

        # average loss across conditionals
        loss = loss / self.hparams.d

        out['log'] = tensorboard_logs
        out['loss'] = loss

        return OrderedDict(out)

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        out = dict(val_loss=avg_loss)

        for j in range(self.hparams.d):
            # condense loss for CDE_j
            avg_loss_j = torch.stack([x[f'loss_{j}'] for x in outputs]).mean()

            # store average validation loss in history
            if 'val_loss' not in self.hparams.history[j]:
                self.hparams.history[j]['val_loss'] = []
            self.hparams.history[j]['val_loss'].append(avg_loss_j)

            tensorboard_logs[f'val_loss_{j}'] = avg_loss_j
            out[f'val_loss_{j}'] = avg_loss_j

        out['log'] = tensorboard_logs

        out['progress_bar'] = {'val_loss': avg_loss}

        return out

    def configure_optimizers(self):

        d = self.hparams.d

        optimizers = []
        schedulers = []
        for j in range(d):
            cde_j = getattr(self, f'cde_{j}')
            optimizer = optim.Adam(cde_j.parameters(),
                                   lr=self.hparams.lr[j],
                                   betas=(0.9, 0.999),
                                   eps=1e-08,
                                   weight_decay=self.hparams.weight_decay[j],
                                   amsgrad=False)

            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=self.hparams.scheduler_patience[j],
                verbose=True,
                threshold=0.05,
                cooldown=self.hparams.scheduler_cooldown[j],
                min_lr=1e-9,
            )
            optimizers.append(optimizer)

            schedulers.append({
                'scheduler': scheduler,
                'monitor': f'val_loss_{j}',  # Default: val_loss
                'interval': 'epoch',
                'frequency': 1
            })

        return optimizers, schedulers
