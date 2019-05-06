"""Datasets for training models in PyTorch."""

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import librosa

from dss.datasets.bandhubpred import BandhubPredset


class BandhubEvalset(BandhubPredset):

    """Class for loading bandhub dataset during evaluation."""

    def __init__(self, metadata, split='train', mag_func='sqrt', n_frames=513,
                 harmonics=False, val_pct=0.1, random_seed=None):
        """
        Initialize BandhubEvalset.

        Args
        ----
            metadata : dataframe, of audio metadata.
            split : string, 'train', 'val' or 'test'.
            concentration : float, concentration param of dirichlet.
            mag_func : string, 'sqrt' or 'log' for magnitude.
            n_frames : int, number of samples in time.
            random_seed : int, random seed to set for temporal sampling.

        """
        BandhubPredset.__init__(
            self, metadata=metadata, split=split, mag_func=mag_func,
            n_frames=n_frames, harmonics=harmonics, val_pct=val_pct,
            random_seed=random_seed)

    @staticmethod
    def _split_track(X, length, dim=1):
        # n_splits = X.size(dim) // length
        slices = list(torch.split(X, length, dim=dim))
        for i, s in enumerate(slices):
            if s.size(dim) < length:
                if dim == 0:
                    if s.dim() == 2:
                        slices[i] = F.pad(s, (0, 0, 0, length - s.size(dim)))
                    else:
                        slices[i] = F.pad(
                            s, (0, length - s.size(dim), 0, 0, 0, 0))
                elif dim == 1:
                    if s.dim() == 2:
                        slices[i] = F.pad(s, (0, length - s.size(dim), 0, 0))
                    else:
                        slices[i] = F.pad(
                            s, (0, 0, 0, length - s.size(dim), 0, 0))
        return torch.stack(slices), torch.tensor([len(slices)])

    @staticmethod
    def collate_func(samples_list):
        """Collate batches."""
        samples = defaultdict(list)
        max_ns = 0
        max_t = 0
        for sample in samples_list:
            max_ns = np.max([max_ns, sample['ns']])
            max_t = np.max([max_t, sample['t']])

        for sample in samples_list:
            if sample['ns'] < max_ns:
                pad_width = max_ns - sample['ns']
                sample['X'] = F.pad(
                    sample['X'], (0, 0, 0, 0, 0, 0, 0, pad_width))
                sample['X_complex'] = F.pad(
                    sample['X_complex'],
                    (0, 0, 0, 0, 0, 0, 0, 0, 0, pad_width))
            if sample['t'] < max_t:
                sample['y_complex'] = F.pad(
                    sample['y_complex'],
                    (0, 0, 0, max_t - sample['t'], 0, 0, 0, 0, 0, 0))

            for k, v in sample.items():
                samples[k].append(v)

        for k, v in samples.items():
            samples[k] = torch.stack(v)

        return samples

    def _load(self, stft_path, volume):
        stft = torch.load(stft_path)
        return torch.mul(stft, volume)

    def __len__(self):
        """Return length of the dataset."""
        return len(self.related_track_idxs)

    def __getitem__(self, i):
        """Return a sample from the dataset."""
        related_track_idxs = self.related_track_idxs.iat[i]

        # load STFT of related stems, sample and add
        for j, track_idx in enumerate(related_track_idxs):
            # load metadata and temp stft tensor
            stft_path = self.metadata.at[track_idx, 'stft_path']
            instrument = self.metadata.at[track_idx, 'instrument']
            volume = self.metadata.at[track_idx, 'trackVolume']
            tmp = self._load(stft_path, volume)

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

        # take magnitude of combined stems and scale and split into chunks
        X_complex = torch.zeros_like(X).copy_(X)
        X = self._stft_mag(X)
        # stack targets and add metadata for collate function
        y_all_complex = torch.stack(y_all_complex)

        if X.dim() == 2:
            X_complex, _ = self._split_track(X_complex, self.n_frames, 1)
            X_complex = X_complex.unsqueeze(1)
            X, ns = self._split_track(X, self.n_frames, 1)
            X = X.unsqueeze(1)
            y_all_complex = y_all_complex.unsqueeze(1)
        else:
            X_complex0, _ = self._split_track(X_complex[0], self.n_frames, 1)
            X_complex1, _ = self._split_track(X_complex[1], self.n_frames, 1)
            X_complex = torch.stack([X_complex0, X_complex1], dim=1)
            X0, _ = self._split_track(X[0], self.n_frames, 1)
            X1, ns = self._split_track(X[1], self.n_frames, 1)
            X = torch.stack([X0, X1], dim=1)

        if self.harmonics:
            X = X.numpy()
            if X.shape[2] == 1025:
                fft_freqs = librosa.fft_frequencies(sr=22050, n_fft=2048)
            elif X.shape[2] == 2049:
                fft_freqs = librosa.fft_frequencies(sr=44100, n_fft=4096)
            X = librosa.interp_harmonics(X, fft_freqs, [1, 2, 3], axis=2)
            X = torch.from_numpy(X)
            X = X.transpose(1, 0).contiguous()
            X = X.view(X.size(0), 6, X.size(3), self.n_frames)

        t = torch.tensor([y_all_complex.size(-2)])
        c = torch.tensor(c_all).long()
        track_idx = torch.tensor(track_idx)

        return {'X': X, 'X_complex': X_complex, 'y_complex': y_all_complex,
                'c': c, 'ns': ns, 't': t, 'track_idx': track_idx}
