"""Datasets for training models in PyTorch."""

import torch
import numpy as np
from dss.datasets.base.basebandhub import BaseBandhub
import librosa


class BandhubPredset(BaseBandhub):

    """Class for loading bandhub dataset during predictions."""

    def __init__(self, metadata, split='train', mag_func='sqrt', n_frames=513,
                 harmonics=False, val_pct=0.1, random_seed=None):
        """
        Initialize BandhubPredset.

        Args
        ----
            metadata : dataframe, of audio metadata.
            split : string, 'train', 'val' or 'test'.
            mag_func : string, 'sqrt' or 'log' for magnitude.
            n_frames : int, number of samples in time.
            random_seed : int, random seed to set for temporal sampling.

        """
        BaseBandhub.__init__(self)
        self.metadata = metadata
        self.split = split
        self.mag_func = mag_func
        self.n_frames = n_frames
        self.random_seed = random_seed
        self.harmonics = harmonics
        self.val_pct = val_pct
        self.target_indexes = None
        self.related_tracks_idxs = None
        self.related_track_instruments = None
        self.n_classes = self.metadata['instrument'].nunique()

        # if self.harmonics:
        #     self.fft_freqs = librosa.fft_frequencies(sr=22050)

        # split data first based on songId
        self._train_test_split()
        self._build_related_tracks()
        self._build_related_instruments()

    def _build_related_instruments(self):
        self.related_track_instruments = self.metadata.groupby(
            'songId')['instrument'].apply(lambda x: list(x))

    def sample(self, p=0.1):
        self.related_track_idxs = self.related_track_idxs.sample(frac=p)

    def __len__(self):
        """Return length of the dataset."""
        return len(self.related_track_idxs)

    def __getitem__(self, i):
        """Return a sample from the dataset."""
        related_track_idxs = self.related_track_idxs.iat[i]

        # set random seed for all sampling
        seed = np.random.randint(0, 10000000)

        # load STFT of  related stems, sample and add
        for j, track_idx in enumerate(related_track_idxs):
            # load metadata and temp stft tensor
            stft_path = self.metadata.at[track_idx, 'stft_path']
            instrument = self.metadata.at[track_idx, 'instrument']
            volume = self.metadata.at[track_idx, 'trackVolume']
            tmp = self._load(stft_path, volume, seed)

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
        if self.harmonics:
            X = X.numpy()
            if X.shape[1] == 1025:
                fft_freqs = librosa.fft_frequencies(sr=22050, n_fft=2048)
            elif X.shape[1] == 2049:
                fft_freqs = librosa.fft_frequencies(sr=44100, n_fft=4096)
            X = librosa.interp_harmonics(X, fft_freqs, [1, 2, 3], axis=1)
            X = torch.from_numpy(X)
            X = X.view(6, X.size(2), self.n_frames)

        # stack targets and add metadata for collate function
        y_all_complex = torch.stack(y_all_complex)

        if X.dim() == 2:
            X_complex = X_complex.unsqueeze(0)
            X = X.unsqueeze(0)
            y_all_complex = y_all_complex.unsqueeze(1)

        t = torch.tensor([y_all_complex.size(-2)])
        c = torch.tensor(c_all).long()
        track_idx = torch.tensor(track_idx)

        return {'X': X, 'X_complex': X_complex, 'y_complex': y_all_complex,
                'c': c, 't': t, 'track_idx': track_idx}
