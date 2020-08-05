import torch
import torch.nn as nn
import torch.optim as optim

from . import gmm
from .. import utils


class ResidualConditional(nn.Module):
    def __init__(self, in_features, n_params=50, hidden_layers=2):
        super().__init__()
        self.hidden_layers = hidden_layers
        self.in_features = in_features
        self.n_params = n_params

        for i in range(self.hidden_layers + 1):
            if i == 0:
                in_features = self.in_features
            else:
                in_features = n_params
            if i == self.hidden_layers:
                out_features = self.in_features
            else:
                out_features = n_params

            fc = nn.Linear(in_features, out_features, bias=False)
            bn = nn.BatchNorm1d(out_features)
            pr_relu = nn.PReLU()

            # fc gets all 0s when initialized
            fc.weight.data.fill_(0.00)

            setattr(self, f'fc_{i+1}', fc)
            setattr(self, f'bn_{i+1}', bn)
            setattr(self, f'pr_relu_{i+1}', pr_relu)

    def forward(self, x):
        z = x

        for i in range(self.hidden_layers + 1):
            fc = getattr(self, f'fc_{i+1}')
            bn = getattr(self, f'bn_{i+1}')
            pr_relu = getattr(self, f'pr_relu_{i+1}')

            z = pr_relu(bn(fc(z)))

        # residual skip connection
        z = z + x

        return z


class DefaultConditional(nn.Module):
    def __init__(self, in_features, out_features, n_params):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_features, n_params, bias=False),
                                 nn.BatchNorm1d(n_params), nn.PReLU(),
                                 nn.Linear(n_params, n_params, bias=False),
                                 nn.BatchNorm1d(n_params), nn.PReLU(),
                                 nn.Linear(n_params, n_params, bias=False),
                                 nn.BatchNorm1d(n_params), nn.PReLU())

    def forward(self, x):
        return self.net(x)


class CDE(nn.Module):
    def __init__(self, hparams):
        """
        hparams must include:
            - d
            - n_components
            - X_mu
            - X_sigma
            - y_mu
            - y_sigma
            - n_params
            - use_safe_softmax
            - init_type
            - hidden_layers
            - j : jth cde (must be passed in if residual initialization)
        """
        super().__init__()

        # attributes
        self.hparams = hparams

        # handle constant features
        self.hparams.X_sigma[self.hparams.X_sigma == 0] = 1
        if self.hparams.y_sigma == 0:
            self.hparams.y_sigma = 1
        # add X_mu, X_sigma, y_mu, y_sigma as parameters of CDE
        self.X_mu = nn.Parameter(self.hparams.X_mu, requires_grad=False)
        self.X_sigma = nn.Parameter(self.hparams.X_sigma, requires_grad=False)
        self.y_mu = nn.Parameter(self.hparams.y_mu, requires_grad=False)
        self.y_sigma = nn.Parameter(self.hparams.y_sigma, requires_grad=False)

        # initialize network layers
        if hasattr(self.hparams, 'init_type'):
            init_type = self.hparams.init_type
        else:
            init_type = 'default'
        if hasattr(self.hparams, 'hidden_layers'):
            hidden_layers = self.hparams.hidden_layers
        else:
            hidden_layers = 2
        self.initialize_layers(init_type, hidden_layers=hidden_layers)

        # initialize history
        self.hparams.history = dict()

    def initialize_layers(self, init_type, hidden_layers):
        if init_type in ['default', 'fixed']:
            self.fc_in = DefaultConditional(self.hparams.d,
                                            self.hparams.n_params,
                                            self.hparams.n_params)

            pi_layer = nn.Linear(self.hparams.n_params, self.hparams.n_components)
            mu_layer = nn.Linear(self.hparams.n_params, self.hparams.n_components)
            sigma_layer = nn.Linear(self.hparams.n_params,
                                    self.hparams.n_components)
            # well-spaced mu layers between [-3, 3] (normalized data)
            mu_layer.weight.data.fill_(0.0)
            mu_layer.bias.data = torch.linspace(-3, 3,
                                                self.hparams.n_components)
            # if fixed init, then change pi and sigma layers
            if init_type == 'fixed':
                # uniform pi
                pi_layer.weight.data.fill_(0.0)
                pi_layer.bias.data = torch.ones(
                    self.hparams.n_components) / self.hparams.n_components

                # std deviation = 1
                sigma_layer.weight.data.fill_(0.0)
                sigma_layer.bias.data = torch.ones(
                    self.hparams.n_components) * torch.log(
                        torch.exp(1 / self.hparams.y_sigma) - 1)

        if init_type == 'residual':
            assert hasattr(self.hparams, 'j'), 'hparams must contain j for residual initialization'
            self.fc_in = ResidualConditional(self.hparams.d,
                                             n_params=self.hparams.n_params,
                                             hidden_layers=hidden_layers)

            pi_layer = nn.Linear(self.hparams.d, self.hparams.n_components)
            mu_layer = nn.Linear(self.hparams.d, self.hparams.n_components)
            sigma_layer = nn.Linear(self.hparams.d, self.hparams.n_components)
            # set mu such that one of the centers is the same as the jth feature
            mu_layer.weight.data = torch.zeros(self.hparams.n_components, self.hparams.d)
            mu_layer.weight.data[0, self.hparams.j] += 1.
            mu_layer.bias.data = torch.cat(
                [torch.zeros(1), torch.linspace(-3, 3, self.hparams.n_components - 1)])
            # uniform pi
            pi_layer.weight.data.fill_(0.0)
            pi_layer.bias.data = torch.ones(
                self.hparams.n_components) / self.hparams.n_components
            

        # define parameter transforms
        ## pi
        self.pi_transform = nn.Sequential(
            pi_layer,
            utils.SafeSoftmax(axis=-1, eps=1e-5)
            if self.hparams.use_safe_softmax else nn.Softmax(dim=-1))
        ## mu
        self.mu_transform = nn.Sequential(mu_layer)
        ## sigma
        self.sigma_transform = nn.Sequential(sigma_layer, nn.Softplus())

    def forward(self, x):
        """
        x must be standardized
        """
        z = self.fc_in(x)

        # get MDN parameters
        pi = self.pi_transform(z.clamp(-1e3, 1e3))
        mu = self.mu_transform(z).clamp(-30, 30)
        sigma = self.sigma_transform(z).clamp(1e-2, 1e2)

        return pi, mu, sigma

    #### Custom functions

    def log_prob(self, x, y, detach_params=False):
        """
        x, y must be un-standardized
        
        y dimension must be 1
        """
        assert len(
            y.shape
        ) == 1, f'y (dimension {y.shape}) must have dimension exactly 1'

        # standardize x and y
        x, y = self.standardize(x, y)

        # get parameters of a gaussian mixture
        pi, mu, sigma = self(x)
        if detach_params:
            pi, mu, sigma = map(lambda var: var.detach(), [pi, mu, sigma])
        q_y = gmm.GaussianMixture(pi, mu, sigma)

        return q_y.log_prob(y) - torch.log(self.hparams.y_sigma)

    def sample(self, x):
        """
        x must be un-standardized
        """
        # standardize x
        x = self.standardize(x)

        # get parameters of a gaussian mixture
        pi, mu, sigma = self(x)
        q_y = gmm.GaussianMixture(pi, mu, sigma)
        standardized_sample = q_y.rsample()

        # return un-standardized sample
        unstandardized_sample = standardized_sample * self.hparams.y_sigma + self.hparams.y_mu
        return unstandardized_sample

    def rsample(self, x):
        """
        x must be un-standardized
        """

        return self.sample(x)

    def standardize(self, x, y=None):
        x = (x - self.X_mu.reshape(1, -1)) / self.X_sigma.reshape(1, -1)
        if y is not None:
            y = (y - self.y_mu) / self.y_sigma
            return x, y
        return x
