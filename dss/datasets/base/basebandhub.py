"""Datasets for training models in PyTorch."""

import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import GroupShuffleSplit

import torch
from torch.distributions.dirichlet import Dirichlet

from dss.datasets.base.baseaudio import BaseAudio


class BaseBandhub(BaseAudio):

    """Base class for loading bandhub dataset."""

    def __init__(self):
        """Initialize BaseBandhub dataset."""
        BaseAudio.__init__(self)
        self.metadata = None
        self.split = None
        self.related_track_idxs = None
        self.concentration = None

    def _build_related_tracks(self):
        self.related_track_idxs = self.metadata.groupby(
            'songId')['trackId'].apply(lambda x: list(x.index.get_values()))

    def _train_test_split(self):
        # create train and val and test splits for artist splitting

        tracks = self.metadata['trackId'].get_values()
        songs = self.metadata['songId'].get_values()

        # train split
        np.random.seed(10)
        tracks, songs = shuffle(tracks, songs)
        gss = GroupShuffleSplit(n_splits=1, test_size=.3, random_state=10)
        train_mask, val_test_mask = next(
            gss.split(X=tracks, y=None, groups=songs))
        train_songs = songs[train_mask]
        val_test_tracks = tracks[val_test_mask]
        val_test_songs = songs[val_test_mask]

        # test and val splits
        gss = GroupShuffleSplit(n_splits=1, test_size=.1/.3, random_state=10)
        test_mask, val_mask = next(
            gss.split(X=val_test_tracks, y=None, groups=val_test_songs))
        val_songs = val_test_songs[val_mask]
        test_songs = val_test_songs[test_mask]

        if self.split == 'train':
            self.metadata = self.metadata[
                self.metadata['songId'].isin(train_songs)]
        elif self.split == 'val':
            self.metadata = self.metadata[
                self.metadata['songId'].isin(val_songs)]
        elif self.split == 'test':
            self.metadata = self.metadata[
                self.metadata['songId'].isin(test_songs)]

    def _sample_volume_alphas(self, n_related):
        dirichlet = Dirichlet(
            torch.tensor([self.concentration for _ in range(n_related)]))
        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)
        return dirichlet.sample()

    def __len__(self):
        """Return length of the dataset."""
        raise NotImplementedError("Not implmented!")

    def __getitem__(self, i):
        """Return a sample from the dataset."""
        raise NotImplementedError("Not implmented!")
