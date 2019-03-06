"""Datasets for training models in PyTorch."""

import torch
import numpy as np
from dss.datasets.base.basebandhub import BaseBandhub


class BandhubPredset(BaseBandhub):

    """Class for loading bandhub dataset during predictions."""

    def __init__(self, metadata, split='train', concentration=50.,
                 random_seed=None):
        """
        Initialize BandhubPredset.

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
        self.target_indexes = None
        self.related_tracks_idxs = None
        self.related_track_instruments = None
        self.n_classes = self.metadata['instrument'].nunique()

        # split data first based on songId
        self._train_test_split()
        self._build_related_tracks()
        self._build_related_instruments()

    def _build_related_instruments(self):
        self.related_track_instruments = self.metadata.groupby(
            'songId')['instrument'].apply(lambda x: list(x))

    def __len__(self):
        """Return length of the dataset."""
        return len(self.related_track_idxs)

    def __getitem__(self, i):
        """Return a sample from the dataset."""
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

            # initialize tensors
            if j == 0:
                X = torch.zeros_like(tmp)
                y_all_complex = [torch.zeros_like(tmp)
                                 for _ in range(self.n_classes)]
                c_all = [0] * self.n_classes

            # add tensors
            X.add_(tmp)
            y_all_complex[instrument].add_(tmp)
            c_all[instrument] = 1

        # take magnitude of combined stems and scale
        X_complex = torch.zeros_like(X).copy_(X)
        X = self._stft_mag(X)

        # stack targets and add metadata for collate function
        y_all_complex = torch.stack(y_all_complex)
        t = torch.tensor([y_all_complex.size(-2)])
        c = torch.tensor(c_all).long()

        return {'X': X, 'X_complex': X_complex, 'y_complex': y_all_complex,
                'c': c, 't': t}
