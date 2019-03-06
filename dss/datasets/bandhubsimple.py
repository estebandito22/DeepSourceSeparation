"""Datasets for training models in PyTorch."""

import torch
import numpy as np
from dss.datasets.base.basebandhub import BaseBandhub


class BandhubDataset(BaseBandhub):

    """Class for loading bandhub dataset."""

    def __init__(self, metadata, split='train', concentration=50.,
                 random_seed=None):
        """
        Initialize BandhubDataset.

        Args
        ----
            metadata : dataframe, of audio metadata.
            split : string, 'train', 'val' or 'test'.
            concentration : float, concentration param of dirichlet.
            random_seed : int, random seed to set for temporal sampling.
        """
        BaseBandhub.__init__(self)
        self.metadata = metadata
        self.split = split
        self.concentration = concentration
        self.random_seed = random_seed
        self.related_tracks_idxs = None
        self.n_classes = self.metadata['instrument'].nunique()

        # split data first based on songId
        self._train_test_split()
        self._build_related_tracks()

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

        # set random seed for all sampling
        seed = np.random.randint(0, 1000)

        # load STFT of  related stems, sample and add
        for j, track_idx in enumerate(related_track_idxs):
            # load metadata and temp stft tensor
            stft_path = self.metadata.at[track_idx, 'stft_path']
            instrument = self.metadata.at[track_idx, 'instrument']
            tmp = self._load(stft_path, volume_alphas[j], seed)

            # initialize input tensor and add
            if j == 0:
                X = torch.zeros_like(tmp)
            X.add_(tmp)

            # magnitdue and scale target for tensors
            tmp = self._stft_mag(tmp * (1/volume_alphas[j])) * volume_alphas[j]

            # initialize target tensor and add
            if j == 0:
                y_all = [torch.zeros_like(tmp) for _ in range(self.n_classes)]
            y_all[instrument].add_(tmp)

        # take magnitude of combined stems and scale
        X = self._stft_mag(X)

        # stack target tensors
        y_all = torch.stack(y_all)

        return {'X': X, 'y': y_all}
