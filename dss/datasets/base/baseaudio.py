"""Datasets for training models in PyTorch."""

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class BaseAudio(Dataset):

    """Base class for loading audio datasets."""

    def __init__(self):
        """Initialize BaseAudio dataset."""

    def _sample(self, X, length, dim=1, random_seed=None):
        if random_seed is not None:
            np.random.seed(random_seed)
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        if X.size()[dim] > length:
            rand_start = np.random.randint(0, X.size()[dim] - length)
        else:
            if dim == 0:
                X = F.pad(X, (0, length - X.size()[dim], 0, 0))
            elif dim == 1:
                X = F.pad(X, (0, 0, 0, length - X.size()[dim]))
            else:
                raise ValueError("dim must be 0 or 1.")
            return X

        if dim == 0:
            X = X[rand_start:rand_start + length]
        elif dim == 1:
            X = X[:, rand_start:rand_start + length]
        else:
            raise ValueError("dim must be 0 or 1.")
        return X

    def _sample_np(self, X, length, dim=1, random_seed=None):
        if random_seed is not None:
            np.random.seed(random_seed)
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        if X.shape[dim] > length:
            rand_start = np.random.randint(0, X.shape[dim] - length)
        else:
            if dim == 0:
                X = np.pad(X, ((0, length - X.shape[dim]), (0, 0)), 'constant')
            elif dim == 1:
                X = np.pad(X, ((0, 0), (0, length - X.shape[dim])), 'constant')
            else:
                raise ValueError("dim must be 0 or 1.")
            return X

        if dim == 0:
            X = X[rand_start:rand_start + length]
        elif dim == 1:
            X = X[:, rand_start:rand_start + length]
        else:
            raise ValueError("dim must be 0 or 1.")
        return X

    def get_batches(self, k=5):
        """Return batches of random song ids."""
        indexes = [x for x in range(len(self))]
        np.random.shuffle(indexes)
        s = 0
        size = int(np.ceil(len(indexes) / k))
        batches = []
        while s < len(indexes):
            batches += [indexes[s:s + size]]
            s = s + size
        return batches

    def _load_transform(self, stft_path, volume, seed):
        stft = torch.load(stft_path)
        if stft.dim() == 4:
            stft0 = self._sample(stft[0], self.n_frames, 1, seed)
            stft1 = self._sample(stft[1], self.n_frames, 1, seed)
            stft = torch.stack([stft0, stft1])
        else:
            stft = self._sample(stft, self.n_frames, 1, seed)
        return self._stft_mag(stft)

    def _load(self, stft_path, volume, seed):
        stft = torch.load(stft_path)
        if stft.dim() == 4:
            stft0 = self._sample(stft[0], self.n_frames, 1, seed)
            stft1 = self._sample(stft[1], self.n_frames, 1, seed)
            stft = torch.stack([stft0, stft1])
        else:
            stft = self._sample(stft, self.n_frames, 1, seed)
        return torch.mul(stft, volume)

    @staticmethod
    def _minmax_scale(x):
        if x.max() - x.min() == 0:
            return torch.zeros_like(x)
        return (x - x.min()) / (x.max() - x.min())

    def _stft_mag(self, x):
        power = torch.sum(x ** 2, dim=-1)
        if self.mag_func == 'sqrt':
            mag = torch.sqrt(power)
        elif self.mag_func == 'log':
            mag = torch.log1p(torch.sqrt(power))
        return mag

    def __len__(self):
        """Return length of the dataset."""
        raise NotImplementedError("Not implmented!")

    def __getitem__(self, i):
        """Return a sample from the dataset."""
        raise NotImplementedError("Not implmented!")
