import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .. import utils


class PiNet(nn.Module):
    def __init__(self, d, init_frac):
        super().__init__()
        self.init_frac = init_frac
        self.d = d

        # initialize swap fraction parameters
        self.log_pi = nn.Parameter(torch.log(
            torch.ones(self.d, 2) * self.init_frac),
                                   requires_grad=True)

    def forward(self, x, x_tilde):
        return self.log_pi


class GumbelSwapper(nn.Module):
    def __init__(self,
                 d,
                 tau,
                 init_frac=[0.5, 0.5],
                 anneal_rate=2e-4,
                 tau_min=0.2,
                 weight_decay=1e-5,
                 adam_betas=(0.9, 0.999), anneal_start_epoch=5):
        # initialize pytoch nn
        super().__init__()

        # initialize hyperparameters
        self.d = d
        self.tau = tau
        self.init_frac = init_frac
        self.tau_min = tau_min
        self.anneal_rate = anneal_rate
        self.weight_decay = weight_decay
        self.adam_betas = adam_betas
        self.anneal_start_epoch = anneal_start_epoch

        # parameters of gumbel softmax
        ## shape = (d, 2)
        init_frac = torch.tensor(init_frac)
        self.pi_net = PiNet(self.d, init_frac)

        # define optimizer for parameters
        self.safe_softmax = utils.SafeSoftmax(axis=2)

        # keep track of epoch
        self.epoch = 0

    def update_tau(self):
        """anneal tau"""
        if self.epoch >= self.anneal_start_epoch:
            self.tau = np.maximum(self.tau * np.exp(-self.anneal_rate * self.epoch),
                                self.tau_min)
        self.epoch += 1

    def forward(self, x, x_tilde):
        """implements straight-through gumbel softmax
        https://arxiv.org/pdf/1611.01144.pdf (Jang et al. 2017)
        """
        n, d = x.shape
        assert d == self.d, f'Dimension of x ({d}) does not match self.d ({self.d}).'
        assert x.shape == x_tilde.shape, 'x and x_tilde have different dimensions.'

        # sample gumbel softmax
        # logits = F.log_softmax(self.log_pi, dim=1)
        gumbels = -torch.empty(n, d, 2).exponential_().log().type_as(x)
        gumbels = (gumbels + self.pi_net(x, x_tilde)) / self.tau
        y_soft = self.safe_softmax(gumbels)

        # straight through gumbel softmax
        index = y_soft.max(axis=2, keepdim=True)[1]
        y_hard = torch.zeros(n, d, 2).type_as(x).scatter_(2, index, 1.0)
        swaps = y_hard - y_soft.detach() + y_soft

        # create swap matrix
        A = swaps[..., 0]
        B = swaps[..., 1]

        u = B * x + A * x_tilde
        u_tilde = A * x + B * x_tilde

        # return swapped x and x_tilde
        return u, u_tilde


class RandomSwapper(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d

    def forward(self, x, x_tilde):
        n, d = x.shape
        assert d == self.d, f'Dimension of x ({d}) does not match self.d ({self.d}).'
        assert x.shape == x_tilde.shape, 'x and x_tilde have different dimensions.'

        bool_swap_inds = np.random.binomial(1, 0.5, size=self.d) == 1
        swap_inds = np.where(bool_swap_inds)[0]
        keep_inds = np.where(~bool_swap_inds)[0]

        u = torch.zeros(n, d)
        u_tilde = torch.zeros(n, d)

        u[:, swap_inds] += x_tilde[:, swap_inds]
        u[:, keep_inds] += x[:, keep_inds]

        u_tilde[:, swap_inds] += x[:, swap_inds]
        u_tilde[:, keep_inds] += x_tilde[:, keep_inds]

        return u, u_tilde

class FixedSwapper(nn.Module):
    def __init__(self, d, swap_inds):
        super().__init__()
        self.d = d
        self.swap_inds = set(list(swap_inds))

    def forward(self, x, x_tilde):
        n, d = x.shape
        assert d == self.d, f'Dimension of x ({d}) does not match self.d ({self.d}).'
        assert x.shape == x_tilde.shape, 'x and x_tilde have different dimensions.'

        swap_inds = np.array([i in self.swap_inds for i in range(d)])
        keep_inds = ~swap_inds

        u = torch.zeros(n, d)
        u_tilde = torch.zeros(n, d)

        u[:, swap_inds] += x_tilde[:, swap_inds]
        u[:, keep_inds] += x[:, keep_inds]

        u_tilde[:, swap_inds] += x[:, swap_inds]
        u_tilde[:, keep_inds] += x_tilde[:, keep_inds]

        return u, u_tilde


if __name__ == "__main__":
    g = GumbelSwapper(d=3, tau=0.1)
    x = torch.tensor(np.random.randint(0, 10, (4, 3)))
    x_tilde = 100 * x
