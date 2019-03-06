"""Datasets for training models in PyTorch."""


import torch
from sklearn.model_selection import train_test_split
from dss.datasets.base.baseaudio import BaseAudio


class MSDDataset(BaseAudio):

    """Class for loading MSD dataset during pretraining."""

    def __init__(self, metadata, split='train', random_seed=None):
        """
        Initialize MSDDataset.

        Args
        ----
            metadata : dataframe, of audio metadata.
            split : string, 'train', 'val' or 'test'.
            random_seed : int, random seed to set for song sampling.

        """
        BaseAudio.__init__(self)
        self.metadata = metadata
        self.split = split
        self.random_seed = random_seed

        # split data first based on songId
        self._train_test_split()

    def _train_test_split(self):
        X_train, X_test = train_test_split(
            self.metadata, test_size=0.2, random_state=10)
        X_train, X_val = train_test_split(
            X_train, test_size=0.05/0.8, random_state=10)

        if self.split == 'train':
            self.metadata = X_train
        elif self.split == 'val':
            self.metadata = X_val
        elif self.split == 'test':
            self.metadata = X_test

    def _load_transform(self, stft_path, volume):
        stft = torch.load(stft_path)
        mag = self._stft_mag(stft)
        if mag.dim() != 2:
            mag = mag.unsqueeze(1)
        return self._sample(mag, 129, 1)

    def __len__(self):
        """Return length of the dataset."""
        return self.metadata.shape[0]

    def __getitem__(self, i):
        """Return a sample from the dataset."""
        # target track metadata
        stft_path = self.metadata.iat[i, 0]

        # load torch target tensor
        X = self._load_transform(stft_path, 1)

        return {'X': X}
