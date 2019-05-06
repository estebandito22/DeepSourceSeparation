"""Datasets for training models in PyTorch."""

import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.uniform import Uniform
from dss.datasets.base.basebandhub import BaseBandhub
from scipy.stats import uniform
import librosa


class BandhubDataset(BaseBandhub):

    """Class for loading bandhub dataset."""

    def __init__(self, metadata, split='train', concentration=1.,
                 mag_func='sqrt', n_frames=513, upper_bound_slope=60,
                 lower_bound_slope=30, threshold=None, chan_swap=0,
                 uniform_volumes=False, interference=0.0, instrument_mask=0.0,
                 gain_slope=0.0, mask_curriculum=0, harmonics=False,
                 val_pct=0.1, random_seed=None):
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
        self.uniform_volumes = uniform_volumes
        self.interference = interference
        self.instrument_mask = instrument_mask
        self.mask_prob = None
        self.mask_curriculum = mask_curriculum
        self.random_seed = random_seed
        self.gain_slope = gain_slope
        self.harmonics = harmonics
        self.val_pct = val_pct
        self.related_tracks_idxs = None
        self.n_classes = self.metadata['instrument'].nunique()

        if not self.mask_curriculum:
            self.mask_prob = self.instrument_mask

        # if self.harmonics:
        #     self.fft_freqs = librosa.fft_frequencies(sr=22050)

        # split data first based on songId
        self._train_test_split()
        self._build_related_tracks()

    def _sample_volume_alphas(self, n_related):
        if self.uniform_volumes:
            u = Uniform(0.25, 1.25)
            return u.sample().repeat(n_related)
        if isinstance(self.concentration, (float, int)):
            concentration = self.concentration
        else:
            concentration = self.concentration.rvs()
        dirichlet = Dirichlet(
            torch.tensor([concentration for _ in range(n_related)]))
        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)
        return dirichlet.sample() * float(self.n_classes)

    def _sample_instrument_mask(self, n_related):
        u = np.random.uniform(size=n_related - 1) <= self.mask_prob
        m = np.random.permutation(
            np.concatenate([np.array([1.0]), u.astype(float)]))
        return torch.from_numpy(m.astype(float)).float()

    def sample(self, p=0.1):
        self.related_track_idxs = self.related_track_idxs.sample(frac=p)

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

    def set_instrument_mask(self, t):
        """Set the instrument mask."""
        self.mask_prob = min(
            (self.instrument_mask - 0.05)/self.mask_curriculum * t + 0.05,
            self.instrument_mask)

    def _chan_swap(self, stft):
        first_chan = int(np.random.uniform() < 1 - self.chan_swap)
        second_chan = 1 - first_chan
        stft = torch.index_select(
            stft, 0, torch.tensor([first_chan, second_chan]))
        return stft

    def _add_interfere(self, X, volume_alphas, seed):
        if np.random.uniform() <= self.interference:
            rand_idx = np.random.randint(0, len(self.related_track_idxs))
            rand_related_track_idxs = self.related_track_idxs.iat[rand_idx]
            rand_related_track_idx = np.random.choice(rand_related_track_idxs)
            rand_related_stft_path = self.metadata.\
                at[rand_related_track_idx, 'stft_path']
            rand_volume_alpha = np.random.choice(volume_alphas)
            rand_related_stft = self._load(
                rand_related_stft_path, rand_volume_alpha, seed)
            return X.add_(rand_related_stft)
        return X

    def _make_mod_func(self, slope_bound, intercept_bound):
        slope = np.random.uniform(-slope_bound, slope_bound)
        intercept = np.random.uniform(-intercept_bound, intercept_bound)
        line1 = torch.from_numpy(
            slope * np.arange(0, self.n_frames) + intercept).float()
        return torch.sigmoid(line1)

    def _gain_modulation(self, stft):
        s_bound = self.gain_slope
        i_bound = self.n_frames / 2 * self.gain_slope

        func1 = self._make_mod_func(s_bound, i_bound)
        func2 = self._make_mod_func(s_bound, i_bound)

        func = torch.max(func1, func2)
        func = func.view(1, 1, self.n_frames, 1)

        return stft * func

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
        if self.instrument_mask:
            volume_alphas *= self._sample_instrument_mask(n_related)

        # set random seed for all sampling
        seed = np.random.randint(0, 10000000)

        # load STFT of  related stems, sample and add
        for j, track_idx in enumerate(related_track_idxs):
            # load metadata and temp stft tensor
            stft_path = self.metadata.at[track_idx, 'stft_path']
            instrument = self.metadata.at[track_idx, 'instrument']
            tmp = self._load(stft_path, volume_alphas[j], seed)
            if self.chan_swap:
                tmp = self._chan_swap(tmp)
            if self.gain_slope:
                tmp = self._gain_modulation(tmp)

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

        # add random interference
        X = self._add_interfere(X, volume_alphas, seed)

        # take magnitude of combined stems and scale
        X = self._stft_mag(X)
        if self.harmonics:
            X = X.numpy()
            if X.shape[1] == 1025:
                fft_freqs = librosa.fft_frequencies(sr=22050, n_fft=2048)
            elif X.shape[1] == 2049:
                fft_freqs = librosa.fft_frequencies(sr=44100, n_fft=4096)
            X = librosa.interp_harmonics(X, fft_freqs, [1, 2, 3], axis=1)
            X = torch.from_numpy(X)
            X = X.view(6, X.size(2), self.n_frames)

        # stack target tensors
        y_all = torch.stack(y_all)

        if X.dim() == 2:
            X = X.unsqueeze(0)
            y_all = y_all.unsqueeze(1)

        return {'X': X, 'y': y_all}
