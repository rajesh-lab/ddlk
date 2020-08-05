import logging

import numpy as np
import torch
from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor, MLPClassifier
from tqdm import tqdm


class KnockoffStatistic(BaseEstimator):
    def __init__(self, cv=None, cv_split=0.8):
        self.cv = cv
        self.cv_split = cv_split

    def compute_statistics(self, X, X_tilde, Y):
        """
        Compute knockoff statistics by training model
        """
        raise NotImplementedError('Method not implemented...')

    def swap(self, X, X_tilde, j=None):
        """
        Returns U.
        U is X with X[:, j] <- X_tilde[:, j]
        """
        assert X.shape == X_tilde.shape, 'X and X_tilde must have the same shape'

        # swap X and X_tilde at index j
        U = X.copy()

        U[:, j] = X_tilde[:, j].copy()

        return U

    def mix(self, X, X_tilde, j, mixture_prop):
        n = len(X)

        rand_idx = np.random.permutation(n)[:int(n * mixture_prop)]
        x_mix = X.copy()
        x_mix[rand_idx, j] = X_tilde[rand_idx, j].copy()
        return x_mix


class HRT_Knockoffs(KnockoffStatistic):
    def __init__(self, mixture_prop=None):
        super().__init__()
        self.mixture_prop = mixture_prop

    def fit(self,
            xTr,
            yTr,
            xTr_tilde=None,
            hidden_layer_sizes=[200],
            alpha=0.01,
            batch_size=64,
            tqdm=tqdm):
        if self.mixture_prop is not None and self.mixture_prop > 0:
            assert xTr_tilde is not None, f'xTr_tilde must be passed into fit for a mixture proportion of [{self.mixture_prop}]'
            assert xTr.shape == xTr_tilde.shape, 'xTr and xTr_tilde do not have the same shape...'
        assert xTr.shape[0] == yTr.shape[
            0], 'xTr and yTr must have the same number of samples'

        to_numpy = lambda x: x.cpu().numpy() if isinstance(x, torch.Tensor
                                                           ) else x
        xTr, xTr_tilde, yTr = map(to_numpy, [xTr, xTr_tilde, yTr])
        n, d = xTr.shape

        # check if mixture statistic is required
        if self.mixture_prop == 0 or self.mixture_prop is None:
            self.model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                                      alpha=alpha,
                                      batch_size=batch_size,
                                      max_iter=1000)
            self.model.fit(xTr, yTr)
        else:
            self.models = dict()

            for j in tqdm(
                    range(d),
                    leave=True,
                    desc=
                    f'Fitting [{d}] mixture proportion [{self.mixture_prop}] statistics...'
            ):
                # mix xTr and xTr_tilde
                xTr_mixture = self.mix(xTr, xTr_tilde, j, self.mixture_prop)
                self.models[j] = MLPRegressor(
                    hidden_layer_sizes=hidden_layer_sizes,
                    alpha=alpha,
                    batch_size=batch_size,
                    max_iter=1000)
                self.models[j].fit(xTr_mixture, yTr)

    def score(self, xTe, yTe, xTe_tilde, tqdm=tqdm):
        assert xTe.shape == xTe_tilde.shape, 'xTe and xTe_tilde do not have the same shape...'
        to_numpy = lambda x: x.cpu().numpy() if isinstance(x, torch.Tensor
                                                           ) else x
        xTe, xTe_tilde, yTe = map(to_numpy, [xTe, xTe_tilde, yTe])

        n, d = xTe.shape
        mse_true = None

        if self.mixture_prop == 0 or self.mixture_prop is None:
            # compute MSE(true)
            y_pred = self.model.predict(xTe)
            mse_true = np.mean((y_pred - yTe)**2)
        else:
            mse_true = dict()
            for j in range(d):
                # compute MSE(true)_j
                y_pred_j = self.models[j].predict(xTe)
                mse_true[j] = np.mean((y_pred_j - yTe)**2)

        self.mse_true = mse_true

        self.null_mse = dict()
        for j in range(d):
            # swap jth feature
            uTe = self.swap(xTe, xTe_tilde, j=j)

            # compute MSE(null)
            if self.mixture_prop == 0 or self.mixture_prop is None:
                y_pred = self.model.predict(uTe)
            else:
                y_pred = self.models[j].predict(uTe)
            # compute MSE with jth feature swapped
            mse_j = np.mean((y_pred - yTe)**2)
            self.null_mse[j] = mse_j

        # compute knockoff statistics
        self.ks = dict()
        for j in range(d):
            if self.mixture_prop == 0 or self.mixture_prop is None:
                self.ks[j] = self.null_mse[j] - self.mse_true
            else:
                self.ks[j] = self.null_mse[j] - self.mse_true[j]

        return self.ks


def mc_entropy(model, x, y):
    out = 0
    yHatProbs = model.predict_proba(x)

    numToRemove = 0
    for i in range(len(y)):
        p_y_x = yHatProbs[i, y[i]]
        if p_y_x == 0:
            numToRemove += 1
        else:
            out -= np.log(p_y_x)

    # warning: this can result in error if numToRemove = len(y)
    return out / (len(y) - numToRemove)


class AMIHRT_Knockoffs(KnockoffStatistic):
    def __init__(self,
                 d,
                 mixture_prop=None,
                 model_class=MLPClassifier,
                 model_args=dict(hidden_layer_sizes=[200],
                                 alpha=0.01,
                                 batch_size=64)):
        super().__init__()
        self.d = d
        self.mixture_prop = mixture_prop
        self.model_class = model_class
        self.model_args = model_args

        if self.mixture_prop is not None and self.mixture_prop > 0:
            self.models = dict()
            for j in range(self.d):
                self.models[j] = self.model_class(**self.model_args)
        else:
            self.model = self.model_class(**self.model_args)

    def fit(self,
            xTr,
            yTr,
            xTr_tilde=None,
            tqdm=tqdm):
        if self.mixture_prop is not None and self.mixture_prop > 0:
            assert xTr_tilde is not None, f'xTr_tilde must be passed into fit for a mixture proportion of [{self.mixture_prop}]'
            assert xTr.shape == xTr_tilde.shape, 'xTr and xTr_tilde do not have the same shape...'
        assert xTr.shape[0] == yTr.shape[
            0], 'xTr and yTr must have the same number of samples'

        to_numpy = lambda x: x.cpu().numpy() if isinstance(x, torch.Tensor
                                                           ) else x
        xTr, xTr_tilde, yTr = map(to_numpy, [xTr, xTr_tilde, yTr])
        n, d = xTr.shape

        # check if mixture statistic is required
        if self.mixture_prop == 0 or self.mixture_prop is None:
            self.model.fit(xTr, yTr)
        else:
            for j in tqdm(
                    range(d),
                    leave=True,
                    desc=
                    f'Fitting [{d}] mixture proportion [{self.mixture_prop}] statistics...'
            ):
                # mix xTr and xTr_tilde
                xTr_mixture = self.mix(xTr, xTr_tilde, j, self.mixture_prop)
                self.models[j].fit(xTr_mixture, yTr)

    def score(self, xTe, yTe, xTe_tilde, test_statistic=mc_entropy, tqdm=tqdm):
        assert xTe.shape == xTe_tilde.shape, 'xTe and xTe_tilde do not have the same shape...'
        to_numpy = lambda x: x.cpu().numpy() if isinstance(x, torch.Tensor
                                                           ) else x
        xTe, xTe_tilde, yTe = map(to_numpy, [xTe, xTe_tilde, yTe])

        n, d = xTe.shape
        ts_true = None

        if self.mixture_prop == 0 or self.mixture_prop is None:
            # compute true test statistic
            ts_true = test_statistic(self.model, xTe, yTe)
        else:
            ts_true = dict()
            for j in range(d):
                # compute jth true test statistic
                ts_true_j = test_statistic(self.models[j], xTe, yTe)
                ts_true[j] = ts_true_j

        self.ts_true = ts_true

        self.null_ts = dict()
        for j in range(d):
            # swap jth feature
            uTe = self.swap(xTe, xTe_tilde, j=j)

            # compute null test statistic with jth feature swapped
            if self.mixture_prop == 0 or self.mixture_prop is None:
                null_ts_j = test_statistic(self.model, uTe, yTe)
            else:
                null_ts_j = test_statistic(self.models[j], uTe, yTe)

            self.null_ts[j] = null_ts_j

        # compute knockoff statistics
        self.ks = dict()
        for j in range(d):
            if self.mixture_prop == 0 or self.mixture_prop is None:
                self.ks[j] = self.null_ts[j] - self.ts_true
            else:
                self.ks[j] = self.null_ts[j] - self.ts_true[j]

        return self.ks


def kfilter(W, offset=0, nominal_fdr=0.1):
    W += offset

    thresholds = np.sort(np.insert(np.abs(W), 0, 0))
    W_matrix = np.stack([W for _ in range(len(thresholds))])
    W_matrix_lt = W_matrix <= -thresholds.reshape(-1, 1)
    W_matrix_gt = W_matrix >= thresholds.reshape(-1, 1)
    fdp_vs_threshold = W_matrix_lt.sum(axis=1) / np.maximum(
        W_matrix_gt.sum(axis=1), 1)

    valid_ts = thresholds[fdp_vs_threshold <= nominal_fdr]

    if len(valid_ts) == 0:
        return np.inf
    return valid_ts[0]


def select(W, offset=0, nominal_fdr=0.1):
    T = kfilter(W, offset=offset, nominal_fdr=nominal_fdr)
    return np.arange(len(W))[W >= T]


def get_fdp_power(W, beta, offset=0, nominal_fdr=0.1):
    T = kfilter(W, offset=offset, nominal_fdr=nominal_fdr)
    predicted_feats = (W >= T).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true=beta.astype(int),
                                      y_pred=predicted_feats).ravel()

    fdp = fp / max(fp + tp, 1.0)
    power = tp / (max(1.0, np.sum(beta)))

    return fdp, power
