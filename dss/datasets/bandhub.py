"""Datasets for training models in PyTorch."""

import torch
import numpy as np
from dss.datasets.base.basebandhub import BaseBandhub


class BandhubDataset(BaseBandhub):

    """Class for loading bandhub dataset."""

    def __init__(self, metadata, split='train', c=0, upsample=True,
                 concentration=50., mag_func='sqrt', random_seed=None):
        """
        Initialize BandhubDataset.

        Args
        ----
            metadata : dataframe, of audio metadata.
            split : string, 'train', 'val' or 'test'.
            c : int, class of target instrument.
            upsample : bool, resample smaller classes to match largest.
            concentration : float, concentration param of dirichlet.
            mag_func : string, 'sqrt' or 'log' for magnitude.
            random_seed : int, random seed to set for temporal sampling.

        """
        BaseBandhub.__init__(self)
        self.metadata = metadata
        self.split = split
        self.c = c
        self.upsample = upsample
        self.concentration = concentration
        self.mag_func = mag_func
        self.random_seed = random_seed
        self.target_indexes = None
        self.related_tracks_idxs = None
        self.n_classes = self.metadata['instrument'].nunique()

        # split data first based on songId
        self._train_test_split()

        # get the length of the data for the largest class and repeat data
        # of the current class to be the same length
        data_len = self.metadata.groupby('instrument').count().max()[0]
        self._build_target_indexes(data_len)

        # filter the metadata to only the target class and build indexes
        self._filter_target_songs()
        self._build_related_tracks()

    def _build_target_indexes(self, data_len):
        mask = self.metadata['instrument'] == self.c
        self.target_indexes = self.metadata[mask].index.get_values()
        if self.upsample:
            tgt_len = len(self.target_indexes)
            if tgt_len < data_len:
                if self.random_seed is not None:
                    np.random.seed(self.random_seed)
                np.random.shuffle(self.target_indexes)
                self.target_indexes = np.repeat(
                    self.target_indexes, data_len // tgt_len + 1)[:data_len]

    def _filter_target_songs(self):
        mask = self.metadata['instrument'] == self.c
        uniq_song_ids = self.metadata[mask]['songId'].unique()
        mask = self.metadata['songId'].isin(uniq_song_ids)
        self.metadata = self.metadata[mask]

    def __len__(self):
        """Return length of the dataset."""
        return len(self.target_indexes)

    def __getitem__(self, i):
        """Return a sample from the dataset."""
        # target track metadata
        idx = self.target_indexes[i]
        songId = self.metadata.at[idx, 'songId']
        related_track_idxs = self.related_track_idxs[songId]

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

        # take magnitude of combined stems
        X = self._stft_mag(X)

        # stack target tensors
        y_all = torch.stack(y_all)

        return {'X': X, 'y': y_all}
