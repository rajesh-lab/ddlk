import torch
from torch.distributions.mixture_same_family import MixtureSameFamily
from .. import utils

class GaussianMixtureRsample(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, pi, mu, sigma):
        """
        X must be [n]-dimensional, not [n x 1]
        """
        ctx.save_for_backward(X, pi, mu, sigma)

        return X

    @staticmethod
    def backward(ctx, grad_output):
        X, pi, mu, sigma = ctx.saved_tensors
        # initialize gradients to be None
        grad_X = grad_pi = grad_mu = grad_sigma = None

        norm = torch.distributions.Normal(mu, sigma)
        # N(X; mu, sigma): [n x K] matrix
        # each column is a different component
        norm_prob = norm.log_prob(X.reshape(-1, 1)).exp()
        # Gaussian CDF: Phi((X - mu) / sigma)
        norm_cum_prob = norm.cdf(X.reshape(-1, 1))
        # denominator; shape = [n x K]
        # where K is the number of mixture components
        mixture_prob = utils.logsumexp(norm_prob.log() + pi.log(),
                                 axis=-1).exp().reshape(-1, 1)

        # check if an input requires gradient
        if ctx.needs_input_grad[0]:
            # grad_X
            grad_X = grad_output.reshape(-1, 1)
        if ctx.needs_input_grad[1]:
            # grad_pi
            numerator = -1 * norm_cum_prob
            grad_pi = grad_output.reshape(-1, 1) * numerator / mixture_prob
        if ctx.needs_input_grad[2]:
            # grad_mu
            numerator = -1 * -pi * norm_prob
            grad_mu = grad_output.reshape(-1, 1) * numerator / mixture_prob
        if ctx.needs_input_grad[3]:
            # grad_sigma
            numerator = -1 * -pi * (X.reshape(-1, 1) - mu) / sigma * norm_prob
            grad_sigma = grad_output.reshape(-1, 1) * numerator / mixture_prob

        return grad_X, grad_pi, grad_mu, grad_sigma


class GaussianMixture(MixtureSameFamily):
    def __init__(self, pi, mu, sigma):
        super().__init__(
            mixture_distribution=torch.distributions.Categorical(pi),
            component_distribution=torch.distributions.Normal(mu, sigma))

    def rsample(self, sample_shape=torch.Size([])):

        X = self.sample(sample_shape)
        pi = self.mixture_distribution.probs
        mu = self.component_distribution.loc
        sigma = self.component_distribution.scale

        return GaussianMixtureRsample.apply(X, pi, mu, sigma)

