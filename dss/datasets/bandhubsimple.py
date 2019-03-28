"""Datasets for training models in PyTorch."""

import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.dirichlet import Dirichlet
from dss.datasets.base.basebandhub import BaseBandhub
from scipy.stats import uniform


class BandhubDataset(BaseBandhub):

    """Class for loading bandhub dataset."""

    def __init__(self, metadata, split='train', concentration=1.,
                 mag_func='sqrt', n_frames=513, upper_bound_slope=60,
                 lower_bound_slope=30, threshold=None, chan_swap=0,
                 random_seed=None):
        """
        Initialize BandhubDataset.

        Args
        ----
            metadata : dataframe, of audio metadata.
            split : string, 'train', 'val' or 'test'.
            concentration : float, concentration param of dirichlet.
            mag_func : string, 'sqrt' or 'log' for magnitude.
            n_frames : int, number of samples in time.
            random_seed : int, random seed to set for temporal sampling.
        """
        BaseBandhub.__init__(self)
        self.metadata = metadata
        self.split = split
        self.concentration = concentration
        self.mag_func = mag_func
        self.n_frames = n_frames
        self.upper_bound_slope = upper_bound_slope
        self.lower_bound_slope = lower_bound_slope
        self.threshold = threshold
        self.chan_swap = chan_swap
        self.random_seed = random_seed
        self.related_tracks_idxs = None
        self.n_classes = self.metadata['instrument'].nunique()

        # split data first based on songId
        self._train_test_split()
        self._build_related_tracks()

    def _sample_volume_alphas(self, n_related):
        if isinstance(self.concentration, (float, int)):
            concentration = self.concentration
        else:
            concentration = self.concentration.rvs()
        dirichlet = Dirichlet(
            torch.tensor([concentration for _ in range(n_related)]))
        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)
        return dirichlet.sample() * float(self.n_classes)

    # def set_concentration(self, t):
    #     """Set the concentration parametr for Dirichlet draws."""
    #     upper_bound = 1e-8 + t * self.upper_bound_slope
    #     lower_bound = 1e-8 + t * self.lower_bound_slope
    #     self.concentration = uniform(lower_bound, upper_bound - lower_bound)

    # def set_concentration(self, t):
    #     """Set the concentration parametr for Dirichlet draws."""
    #     upper_bound = np.power(10, t * self.upper_bound_slope/1000 - 3)
    #     lower_bound = np.power(10, t * self.lower_bound_slope/1000 - 3)
    #     self.concentration = uniform(lower_bound, upper_bound - lower_bound)

    def set_concentration(self, t):
        """Set the concentration parametr for Dirichlet draws."""
        upper_bound = np.power(1.5, t * self.upper_bound_slope/1000 - 5) - (np.power(1.5, -5) - 0.01)
        lower_bound = np.power(1.5, t * self.lower_bound_slope/1000 - 5) - (np.power(1.5, -5) - 0.01)
        self.concentration = uniform(lower_bound, upper_bound - lower_bound)

    def _chan_swap(self, stft):
        first_chan = int(np.random.uniform() < 1 - self.chan_swap)
        second_chan = 1 - first_chan
        stft = torch.index_select(
            stft, 0, torch.tensor([first_chan, second_chan]))
        return stft

    def __len__(self):
        """Return length of the dataset."""
        return len(self.related_track_idxs)

    def __getitem__(self, i):
        """Return a sample from the dataset."""
        # target track metadata
        related_track_idxs = self.related_track_idxs.iat[i]

        # sample volume alphas
        n_related = len(related_track_idxs)
        volume_alphas = self._sample_volume_alphas(n_related)
        if self.threshold:
            volume_alphas = F.threshold(volume_alphas, self.threshold, 0.0)

        # set random seed for all sampling
        seed = np.random.randint(0, 1000)

        # load STFT of  related stems, sample and add
        for j, track_idx in enumerate(related_track_idxs):
            # load metadata and temp stft tensor
            stft_path = self.metadata.at[track_idx, 'stft_path']
            instrument = self.metadata.at[track_idx, 'instrument']
            tmp = self._load(stft_path, volume_alphas[j], seed)
            if self.chan_swap:
                tmp = self._chan_swap(tmp)

            # initialize input tensor and add
            if j == 0:
                X = torch.zeros_like(tmp)
            X.add_(tmp)

            # magnitdue and scale target for tensors
            tmp = self._stft_mag(tmp)

            # initialize target tensor and add
            if j == 0:
                y_all = [torch.zeros_like(tmp) for _ in range(self.n_classes)]
            y_all[instrument].add_(tmp)

        # take magnitude of combined stems and scale
        X = self._stft_mag(X)

        # stack target tensors
        y_all = torch.stack(y_all)

        if X.dim() == 2:
            X = X.unsqueeze(0)
            y_all = y_all.unsqueeze(1)

        return {'X': X, 'y': y_all}
