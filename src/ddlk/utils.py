import logging
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm


def get_gpu(minimum_mb=2000):
    if torch.cuda.is_available():
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            visible_devices = os.environ['CUDA_VISIBLE_DEVICES']
            logging.debug(f'CUDA_VISIBLE_DEVICES: {visible_devices}')
        else:
            visible_devices = None

        if visible_devices is None:
            logging.debug(f'No CUDA_VISIBLE_DEVICES environment variable...')
            os.system(
                'nvidia-smi -q -d Memory | grep -A4 GPU | grep Free > /tmp/gpu-stats'
            )
        else:
            logging.debug(f'CUDA_VISIBLE_DEVICES={visible_devices} found...')
            os.system(
                f'CUDA_VISIBLE_DEVICES="{visible_devices}" nvidia-smi -q -d Memory | grep -A4 GPU | grep Free > /tmp/gpu-stats'
            )
        memory_available = np.array([
            int(x.split()[2]) for x in open('/tmp/gpu-stats', 'r').readlines()
        ])
        os.system('rm -f /tmp/gpu-stats')

        logging.debug(f'Free memory per device: {memory_available}')

        if np.any(memory_available >= minimum_mb):
            logging.debug(f'Using device cuda:{np.argmax(memory_available)}')
            return torch.device(f'cuda:{np.argmax(memory_available)}')
        else:
            logging.debug(f'No free GPU device found. Using cpu...')
            return torch.device('cpu')
    logging.debug(f'No GPU device found. Using cpu...')
    return torch.device('cpu')


def create_folds(X, k):
    if isinstance(X, int) or isinstance(X, np.integer):
        indices = np.arange(X)
    elif hasattr(X, '__len__'):
        indices = np.arange(len(X))
    else:
        indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    folds = []
    start = 0
    end = 0
    for f in range(k):
        start = end
        end = start + len(indices) // k + (1 if (len(indices) % k) > f else 0)
        folds.append(indices[start:end])
    return folds


def batches(indices, batch_size, shuffle=True):
    order = np.copy(indices)
    if shuffle:
        np.random.shuffle(order)
    nbatches = int(np.ceil(len(order) / float(batch_size)))
    for b in range(nbatches):
        idx = order[b * batch_size:min((b + 1) * batch_size, len(order))]
        yield idx


def logsumexp(inputs, dim=None, keepdim=False, axis=None):
    """Numerically stable logsumexp.

    Args:
        inputs: A Variable with any shape.
        dim: An integer.
        keepdim: A boolean.

    Returns:
        Equivalent of log(sum(exp(inputs), dim=dim, keepdim=keepdim)).

    Taken from https://github.com/pytorch/pytorch/issues/2591
    """
    # For a 1-D array x (any array along a single dimension),
    # log sum exp(x) = s + log sum exp(x - s)
    # with s = max(x) being a common choice.

    if axis is not None:
        dim = axis

    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


def create_dataloader(*arr, batch_size=64, shuffle=True, drop_last=False):
    """
    Creates pytorch data loaders from numpy arrays
    """
    train_tensors = [torch.from_numpy(a).float() for a in arr]
    train_dataset = torch.utils.data.TensorDataset(*train_tensors)

    return torch.utils.data.DataLoader(train_dataset,
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                       drop_last=drop_last)


class JointDataset(Dataset):
    def __init__(self, X, Y=None, beta=None, d=None, mode='generation'):
        self.X = X
        self.Y = Y
        self.beta = beta
        self.set_d(d)  # initial stage
        self.mode = mode

        # generation returns only xs
        # prediction returns xs and ys
        assert self.mode in ['generation', 'prediction']

    def __getitem__(self, index):
        if self.mode == 'generation':
            x = self.X[index]
            if self.d is None:
                return tuple([x])
            else:
                if self.d == 0:
                    return tuple([torch.tensor([1.0]), x[0]])
                else:
                    return tuple([x[:self.d], x[self.d]])
        elif self.mode == 'prediction':
            assert self.Y is not None, 'Y cannot be None...'
            x = self.X[index]
            y = self.Y[index]
            return tuple([x, y])
        else:
            raise NotImplementedError(
                f'Data loader mode [{self.mode}] is not implemented...')

    def set_mode(self, mode='generation'):
        """Outputs label data in addition to training data
        x data is all columns of X, y data is Y
        """
        self.mode = mode

    def set_d(self, d=None):
        """Chooses dimension to train on:
            d = 0 -> x data is just 1s, y data is first column of X
            d = 1 -> x data is 1st column of X, y data is second column
            d = 2 -> x data is 1st 2 columns of X, y data is third column

            d must be in the range [0, X.shape[-1] - 1]
        """
        self.d = d

    def reset(self):
        """Resets JointDataset to original state"""
        self.set_mode()
        self.set_d()

    def __len__(self):
        return len(self.X)


def create_jointloader(X,
                       Y=None,
                       beta=None,
                       batch_size=64,
                       shuffle=True,
                       drop_last=False):
    """
    Create pytorch data loader from numpy array.
    Can split data appropriately for complete conditionals
    using the `.set_d()` method.
    Can also split data for prediction tasks for X -> Y
    Stores true feature importance `beta`
    """
    jd = JointDataset(X=X, Y=Y, beta=beta, mode='generation')
    return torch.utils.data.DataLoader(jd,
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                       drop_last=drop_last)


def get_two_moments(dataloader, safe=True, unbiased=False):
    running_sum = None
    running_counts = None

    for lst in dataloader:
        if running_sum is None:
            tmp_ct = []
            tmp_sum = []
            for elem in lst:
                tmp_ct.append(len(elem))
                tmp_sum.append(elem.sum(axis=0))

            running_sum = tmp_sum
            running_counts = tmp_ct
        else:
            for i, elem in enumerate(lst):
                running_sum[i] += elem.sum(axis=0)
                running_counts[i] += len(elem)

    running_means = [(s / c) for s, c in zip(running_sum, running_counts)]

    # doesn't yet have normalization
    running_var = None
    for lst in dataloader:
        if running_var is None:
            running_var = [((elem - running_means[i])**2).sum(axis=0)
                           for i, elem in enumerate(lst)]
        else:
            for i, elem in enumerate(lst):
                running_var[i] += ((elem - running_means[i])**2).sum(axis=0)

    if unbiased:
        offset = 1
    else:
        offset = 0
    running_stds = [(s / (c - offset))**0.5
                    for s, c in zip(running_var, running_counts)]

    if safe:
        for elem in running_stds:
            elem[elem == 0] = 1.0

    return running_means, running_stds


def extract_data(dataloader):
    out = None

    for lst in dataloader:
        if out is None:
            out = lst
        else:
            tmp = [torch.cat([a, b], axis=0) for a, b in zip(out, lst)]
            out = tmp

    return out


class SafeSoftmax(nn.Module):
    def __init__(self, axis=-1, eps=1e-5):
        """
        Safe softmax class
        """
        super().__init__()
        self.axis = axis
        self.eps = eps

    def forward(self, x):
        """
        apply safe softmax in 
        """

        e_x = torch.exp(x - torch.max(x, axis=self.axis, keepdims=True)[0])
        p = e_x / torch.sum(e_x, axis=self.axis, keepdims=True) + self.eps
        p_sm = p / torch.sum(p, axis=self.axis, keepdims=True)

        return p_sm

class Hyperparameters(object):
    def __init__(self, input_dict):
        for key in input_dict.keys():
            if key not in {'self'}:
                setattr(self, key, input_dict[key])
                if key in {'kw_args', 'kwargs'}:
                    for kw_arg in input_dict[key].keys():
                        setattr(self, kw_arg, input_dict[key][kw_arg])

    def __repr__(self):
        elems = ', '.join([f'{key}={self.__dict__[key]}' for key in self.__dict__.keys()])
        return f'Hyperparameters({elems})'
