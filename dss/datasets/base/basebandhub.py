"""Datasets for training models in PyTorch."""

import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import GroupShuffleSplit

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
        self.metadata = self.metadata.copy()
        # create train and val and test splits for artist splitting
        if 'split' in self.metadata.columns:
            train_songs = self.metadata[
                self.metadata['split'] == 'train']['songId'].unique()
            val_songs = self.metadata[
                self.metadata['split'] == 'val']['songId'].unique()
            test_songs = self.metadata[
                self.metadata['split'] == 'test']['songId'].unique()

            if len(val_songs) == 0:

                tracks = self.metadata[
                    self.metadata['songId'].isin(
                        train_songs)]['trackId'].get_values()
                songs = self.metadata[
                    self.metadata['songId'].isin(
                        train_songs)]['songId'].get_values()

                # test split
                np.random.seed(10)
                tracks, songs = shuffle(tracks, songs)

                gss = GroupShuffleSplit(
                    n_splits=1, test_size=0.1, random_state=10)
                train_mask, val_mask = next(
                    gss.split(X=tracks, y=None, groups=songs))
                val_songs = songs[val_mask]
                train_songs = songs[train_mask]

        else:
            tracks = self.metadata['trackId'].get_values()
            songs = self.metadata['songId'].get_values()

            # test split
            np.random.seed(10)
            tracks, songs = shuffle(tracks, songs)
            gss = GroupShuffleSplit(
                n_splits=1, test_size=.2, random_state=10)
            train_val_mask, test_mask = next(
                gss.split(X=tracks, y=None, groups=songs))
            test_songs = songs[test_mask]
            train_val_tracks = tracks[train_val_mask]
            train_val_songs = songs[train_val_mask]

            # train and val splits
            gss = GroupShuffleSplit(
                n_splits=1, test_size=.1/.8, random_state=10)
            train_mask, val_mask = next(
                gss.split(X=train_val_tracks, y=None, groups=train_val_songs))
            val_songs = train_val_songs[val_mask]
            train_songs = train_val_songs[train_mask]

        if self.split == 'train':
            self.metadata = self.metadata[
                self.metadata['songId'].isin(train_songs)]
        elif self.split == 'val':
            self.metadata = self.metadata[
                self.metadata['songId'].isin(val_songs)]
        elif self.split == 'test':
            self.metadata = self.metadata[
                self.metadata['songId'].isin(test_songs)]

    def __len__(self):
        """Return length of the dataset."""
        raise NotImplementedError("Not implmented!")

    def __getitem__(self, i):
        """Return a sample from the dataset."""
        raise NotImplementedError("Not implmented!")
