"""Utiliy functions."""

import warnings
from copy import deepcopy
from collections import defaultdict

import librosa
from scipy.signal import stft, istft
import museval

from collections import namedtuple

from numpy import (isscalar, r_, log, around, unique, asarray,
                   zeros, arange, sort, amin, amax, any, atleast_1d,
                   sqrt, ceil, floor, array, compress,
                   pi, exp, ravel, count_nonzero, sin, cos, arctan2, hypot)

from scipy import stats
from scipy.stats import find_repeats, distributions

import torch
import torch.nn.functional as F
import numpy as np


def torch_correlate(a, kernel_size):
    """
    Perform a cross correlation.

    Parameters
    ----------
    a : tensor, [N, 1, H, W]
    kernel_size : int, kernel size for window.  must be odd.

    Returns
    -------
    out : tensor
        cross correlated tensor with the same shape as 'a'.

    """
    bs, in_channels, height, width = a.size()
    a_unfold = F.unfold(a, kernel_size=kernel_size, padding=kernel_size // 2)
    kernels_flat = torch.ones((bs, in_channels, a_unfold.size(1)))
    if a.is_cuda:
        kernels_flat = kernels_flat.cuda()
    res = kernels_flat.matmul(a_unfold)

    return res.view(bs, in_channels, height, width)


def torch_wiener(im, mysize=None, noise=None):
    """
    Perform a Wiener filter on an N-dimensional array.
    Apply a Wiener filter to the N-dimensional array `im`.
    Parameters
    ----------
    im : tensor
        shappe [N, 1, H, W].
    mysize : int, optional
        A scalar giving the size of the Wiener filter
        window in each dimension.  Elements of mysize should be odd.
    noise : tensor, optional
        The noise-power to use. If None, then noise is estimated as the
        average of the local variance of the input.
    Returns
    -------
    out : tensor
        Wiener filtered result with the same shape as `im`.

    """
    if mysize is None:
        mysize = torch.tensor([3] * im.dim()).float()
    else:
        mysize = torch.tensor([mysize] * im.dim()).float()

    zero_tensor = torch.tensor([0]).float()
    one_tensor = torch.tensor([1]).float()
    if im.is_cuda:
        mysize = mysize.cuda()
        zero_tensor = zero_tensor.cuda()
        one_tensor = one_tensor.cuda()

    # Estimate the local mean
    lMean = torch_correlate(im, int(mysize[0])) / torch.prod(mysize)

    # Estimate the local variance
    lVar = (torch_correlate(im ** 2, int(mysize[0])) / torch.prod(mysize)
            - lMean ** 2)

    # Estimate the noise power if needed.
    if noise is None:
        noise = torch.mean(lVar.view(lVar.size(0), -1), dim=1)
        noise = noise.view(noise.size(0), 1, 1, 1)

    res = (im - lMean)
    res *= (1 - noise / torch.where(lVar == zero_tensor, one_tensor, lVar))
    res += lMean
    out = torch.where(lVar < noise, lMean, res)

    return out


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def mwf(estimates, mixture, kernel_size=3, noise=0.25):
    p = kernel_size // 2

    v_hat = 1./2 * np.sum(estimates * np.conj(estimates), axis=1)

    a = np.expand_dims(estimates.transpose([0, 2, 3, 1]), -1)
    b = np.expand_dims(estimates.transpose([0, 2, 3, 1]), -2)
    R_num = np.matmul(a, np.conj(b))
    R_num = R_num.transpose([0, 1, 3, 4, 2])
    R_num = np.pad(R_num, ((0, 0), (0, 0), (0, 0), (0, 0), (p, p)), 'constant')
    R_num = np.sum(rolling_window(R_num, kernel_size), -1)
    R_num = R_num.transpose([0, 1, 4, 2, 3])

    R_den = np.pad(v_hat, ((0, 0), (0, 0), (p, p)), 'constant')
    R_den = np.sum(rolling_window(R_den, kernel_size), -1)
    R_den = np.expand_dims(np.expand_dims(R_den, -1), -1)
    R_den = np.where(R_den < 1e-20, 1e-20, R_den)

    R_den_noise = np.pad(v_hat, ((0, 0), (0, 0), (p, p)), 'constant')
    R_den_noise = np.sum(rolling_window(R_den_noise, kernel_size), -1)
    R_den_noise = np.expand_dims(np.expand_dims(R_den_noise, -1), -1)
    R_den_noise = np.where(R_den < 1e-20, 1e-20, R_den)

    R_hat = R_num / R_den
    P = np.expand_dims(np.expand_dims(v_hat, -1), -1) * R_hat
    if noise:
        P += np.expand_dims(np.mean(np.expand_dims(
            np.expand_dims(v_hat, -1), -1), -3), -3) * noise
    P_sum = np.sum(P, axis=0)
    P_sum_inv = np.linalg.pinv(P_sum)
    W = np.matmul(P, P_sum_inv)

    mixture = mixture.transpose([0, 2, 3, 1])
    mixture = np.expand_dims(mixture, -1)
    S_hat = np.matmul(W, mixture)
    S_hat = np.squeeze(S_hat, -1).transpose([0, 3, 1, 2])

    return S_hat


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]


def medians(datas, interpolate=False):
    """
    Get the median across songs of mean values by frame.

    Parameters
    ----------
        datas: list, list of dictionary of framewise metrics.

    Returns
    -------
    str
        median values of all target metrics across songs.

    """
    out = defaultdict(list)
    for data in datas:
        if hasattr(data, 'scores'):
            data = data.scores
        for t in data['targets']:
            y = np.array([np.float(f['metrics']['SDR']) for f in t['frames']])
            # trimmed_y = np.trim_zeros(np.where(np.isnan(y), 0, y))
            # y = np.where(trimmed_y == 0, np.nan, trimmed_y)
            if interpolate:
                nans, x = nan_helper(y)
                y[nans] = np.interp(x(nans), x(~nans), y[~nans])
            out[t['name']] += [np.nanmean(y)]

    for k, v in out.items():
        out[k] = np.median(v)

    return out


def means(datas, interpolate=False):
    """
    Get the median across songs of mean values by frame.

    Parameters
    ----------
        datas: list, list of dictionary of framewise metrics.

    Returns
    -------
    str
        median values of all target metrics across songs.

    """
    out = defaultdict(list)
    for data in datas:
        if hasattr(data, 'scores'):
            data = data.scores
        for t in data['targets']:
            y = np.array([np.float(f['metrics']['SDR']) for f in t['frames']])
            # trimmed_y = np.trim_zeros(np.where(np.isnan(y), 0, y))
            # y = np.where(trimmed_y == 0, np.nan, trimmed_y)
            if interpolate:
                nans, x = nan_helper(y)
                y[nans] = np.interp(x(nans), x(~nans), y[~nans])
            out[t['name']] += [np.nanmean(y)]

    for k, v in out.items():
        out[k] = np.mean(v)

    return out


def song_means(datas, interpolate=False):
    """
    Get the median across songs of mean values by frame.

    Parameters
    ----------
        datas: list, list of dictionary of framewise metrics.

    Returns
    -------
    str
        median values of all target metrics across songs.

    """
    out = defaultdict(list)
    for data in datas:
        if hasattr(data, 'scores'):
            data = data.scores
        for t in data['targets']:
            y = np.array([np.float(f['metrics']['SDR']) for f in t['frames']])
            # trimmed_y = np.trim_zeros(np.where(np.isnan(y), 0, y))
            # y = np.where(trimmed_y == 0, np.nan, trimmed_y)
            if interpolate:
                nans, x = nan_helper(y)
                y[nans] = np.interp(x(nans), x(~nans), y[~nans])
            out[t['name']] += [np.nanmean(y)]

    return out


def get_metric_array(data, metric, interpolate=False):
    """
    Get the framewise metrics.

    Parameters
    ----------
        data: dict, dictionary of framewise metrics for each target.
        metric: 'SDR', 'SAR', 'SIR'.

    Returns
    -------
    str
        framewise metrics of all targets in data.

    """
    out = defaultdict(list)
    for t in data['targets']:
        y = np.array([np.float(f['metrics']['SDR']) for f in t['frames']])
        # trimmed_y = np.trim_zeros(np.where(np.isnan(y), 0, y))
        # y = np.where(trimmed_y == 0, np.nan, trimmed_y)
        if interpolate:
            nans, x = nan_helper(y)
            y[nans] = np.interp(x(nans), x(~nans), y[~nans])
        out[t['name']] = np.array(y)

    return out


def combine_metric_arrays(datas, metric, interpolate=False):
    """
    Get the median across songs of mean values by frame.

    Parameters
    ----------
        datas: list, list of dictionary of framewise metrics.
        metric: 'SDR', 'SAR', 'SIR'.

    Returns
    -------
    str
        framewise metrics of all targets in datas stacked together.

    """
    out = defaultdict(list)
    for data in datas:
        metric_dict = get_metric_array(data, metric, interpolate)
        for k, v in metric_dict.items():
            out[k] = np.concatenate([out[k], v])
    return out


def IBM(track, alpha=1, theta=0.5, eval_dir=None):
    """Ideal Binary Mask:
    processing all channels inpependently with the ideal binary mask.
    the mix is send to some source if the spectrogram of that source over that
    of the mix is greater than theta, when the spectrograms are take as
    magnitude of STFT raised to the power alpha. Typical parameters involve a
    ratio of magnitudes (alpha=1) and a majority vote (theta = 0.5)
    """

    # parameters for STFT
    nfft = 2048

    # small epsilon to avoid dividing by zero
    eps = np.finfo(np.float).eps

    # compute STFT of Mixture
    N = track.audio.shape[0]  # remember number of samples for future use
    X = stft(track.audio.T, nperseg=nfft)[-1]
    (I, F, T) = X.shape

    # perform separtion
    estimates = {}
    accompaniment_source = 0
    for name, source in track.sources.items():

        # compute STFT of target source
        Yj = stft(source.audio.T, nperseg=nfft)[-1]

        # Create Binary Mask
        Mask = np.divide(np.abs(Yj)**alpha, (eps + np.abs(X)**alpha))
        Mask[np.where(Mask >= theta)] = 1
        Mask[np.where(Mask < theta)] = 0

        # multiply mask
        Yj = np.multiply(X, Mask)

        # inverte to time domain and set same length as original mixture
        target_estimate = istft(Yj)[1].T[:N, :]

        # set this as the source estimate
        estimates[name] = target_estimate

        # accumulate to the accompaniment if this is not vocals
        if name != 'vocals':
            accompaniment_source += target_estimate

    # set accompaniment source
    estimates['accompaniment'] = accompaniment_source

    data = museval.eval_mus_track(
        track,
        estimates,
        output_dir=eval_dir,
    )

    return estimates, data


def IRM(track, alpha=2, eval_dir=None):
    """Ideal Ratio Mask:
    processing all channels inpependently with the ideal ratio mask.
    this is the ratio of spectrograms, where alpha is the exponent to take for
    spectrograms. usual values are 1 (magnitude) and 2 (power)"""

    # STFT parameters
    nfft = 2048

    # small epsilon to avoid dividing by zero
    eps = np.finfo(np.float).eps

    # compute STFT of Mixture
    N = track.audio.shape[0]  # remember number of samples for future use
    X = stft(track.audio.T, nperseg=nfft)[-1]
    (I, F, T) = X.shape

    # Compute sources spectrograms
    P = {}
    # compute model as the sum of spectrograms
    model = eps

    for name, source in track.sources.items():
        # compute spectrogram of target source:
        # magnitude of STFT to the power alpha
        P[name] = np.abs(stft(source.audio.T, nperseg=nfft)[-1])**alpha
        model += P[name]

    # now performs separation
    estimates = {}
    accompaniment_source = 0
    for name, source in track.sources.items():
        # compute soft mask as the ratio between source spectrogram and total
        Mask = np.divide(np.abs(P[name]), model)

        # multiply the mix by the mask
        Yj = np.multiply(X, Mask)

        # invert to time domain
        target_estimate = istft(Yj)[1].T[:N, :]

        # set this as the source estimate
        estimates[name] = target_estimate

        # accumulate to the accompaniment if this is not vocals
        if name != 'vocals':
            accompaniment_source += target_estimate

    estimates['accompaniment'] = accompaniment_source

    if eval_dir is not None:
        museval.eval_mus_track(
            track,
            estimates,
            output_dir=eval_dir,
        )

    data = museval.eval_mus_track(
        track,
        estimates,
        output_dir=eval_dir,
    )

    return estimates, data


class MedleyStem(object):

    """Simple wrapper for medley audio to work with IBM."""

    def __init__(self, audio):
        """Initialize MedleyStem."""
        self.audio = audio.T


class MedleyTrack(object):

    """Simple wrapper for medley track to work with IBM."""

    def __init__(self, medleydb_track):
        """Initialize MedleyTrack."""
        self.orig_track_obj = medleydb_track
        self.top_instruments = ['electric guitar', 'drum set', 'electric bass',
                                'male singer', 'piano', 'fx/processed sound',
                                'vocalists', 'female singer']
        self.rate = 41000

        self.audio = None
        self.sources = None
        self.targets = None

        self._set_sources()
        self._set_targets()
        self._set_audio()

    def _set_audio(self):
        y = 0.
        for source in self.sources.values():
            y += source.audio
        self.audio = y
        # y, _ = librosa.load(self.orig_track_obj.mix_path, sr=41000, mono=False)
        # self.audio = y.T

    def _set_sources(self):
        stem_dict = defaultdict(list)
        instruments = self.orig_track_obj.stem_instruments
        file_paths = self.orig_track_obj.stem_filepaths()
        mx_coefficients = self.orig_track_obj.mixing_coefficients['audio'].values()
        for name, fp, mx in zip(instruments, file_paths, mx_coefficients):
            if name.find('electric guitar') > -1:
                name = 'electric guitar'
            if name not in self.top_instruments:
                name = 'other'
            y, _ = librosa.load(fp, sr=41000, mono=False)
            y *= mx
            if name in stem_dict:
                stem_dict[name] += y
            else:
                stem_dict[name] = y

        for k, v in stem_dict.items():
            stem_dict[k] = MedleyStem(v)

        self.sources = stem_dict

    def _set_targets(self):
        self.targets = deepcopy(self.sources)


class MusTrack(object):

    """Wrapper for reconstructed audio to fit with museval."""

    def __init__(self, target_recons, track, rate=41000):
        print("Target recons bass", target_recons['bass'], flush=True)
        """Initialize MusTrack."""
        self.audio = None
        for v in target_recons.values():
            if self.audio is None:
                self.audio = v
            else:
                self.audio += v

        self.rate = rate
        self.name = track.name
        self.subset = track.subset

        self.sources = {k: MedleyStem(v.T) for k, v in target_recons.items()}
        self.targets = {k: MedleyStem(v.T) for k, v in target_recons.items()}


class MedleyTrackWrap(object):

    def __init__(self, name):
        self.name = name
        self.subset = test



WilcoxonResult = namedtuple('WilcoxonResult', ('statistic', 'pvalue', 'return_type', 'return_val'))

def wilcoxon(x, y=None, zero_method="wilcox", correction=False):
    """
    Calculate the Wilcoxon signed-rank test.
    The Wilcoxon signed-rank test tests the null hypothesis that two
    related paired samples come from the same distribution. In particular,
    it tests whether the distribution of the differences x - y is symmetric
    about zero. It is a non-parametric version of the paired T-test.
    Parameters
    ----------
    x : array_like
        The first set of measurements.
    y : array_like, optional
        The second set of measurements.  If `y` is not given, then the `x`
        array is considered to be the differences between the two sets of
        measurements.
    zero_method : string, {"pratt", "wilcox", "zsplit"}, optional
        "pratt":
            Pratt treatment: includes zero-differences in the ranking process
            (more conservative)
        "wilcox":
            Wilcox treatment: discards all zero-differences
        "zsplit":
            Zero rank split: just like Pratt, but splitting the zero rank
            between positive and negative ones
    correction : bool, optional
        If True, apply continuity correction by adjusting the Wilcoxon rank
        statistic by 0.5 towards the mean value when computing the
        z-statistic.  Default is False.
    Returns
    -------
    statistic : float
        The sum of the ranks of the differences above or below zero, whichever
        is smaller.
    pvalue : float
        The two-sided p-value for the test.
    Notes
    -----
    Because the normal approximation is used for the calculations, the
    samples used should be large.  A typical rule is to require that
    n > 20.
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test
    """

    if zero_method not in ["wilcox", "pratt", "zsplit"]:
        raise ValueError("Zero method should be either 'wilcox' "
                         "or 'pratt' or 'zsplit'")

    if y is None:
        d = asarray(x)
    else:
        x, y = map(asarray, (x, y))
        if len(x) != len(y):
            raise ValueError('Unequal N in wilcoxon.  Aborting.')
        d = x - y

    if zero_method == "wilcox":
        # Keep all non-zero differences
        d = compress(np.not_equal(d, 0), d, axis=-1)

    count = len(d)
    if count < 10:
        warnings.warn("Warning: sample size too small for normal approximation.")

    r = stats.rankdata(abs(d))
    r_plus = np.sum((d > 0) * r, axis=0)
    r_minus = np.sum((d < 0) * r, axis=0)

    if zero_method == "zsplit":
        r_zero = np.sum((d == 0) * r, axis=0)
        r_plus += r_zero / 2.
        r_minus += r_zero / 2.

    T = min(r_plus, r_minus)
    mn = count * (count + 1.) * 0.25
    se = count * (count + 1.) * (2. * count + 1.)

    if zero_method == "pratt":
        r = r[d != 0]

    replist, repnum = find_repeats(r)
    if repnum.size != 0:
        # Correction for repeated elements.
        se -= 0.5 * (repnum * (repnum * repnum - 1)).sum()

    se = sqrt(se / 24)
    correction = 0.5 * int(bool(correction)) * np.sign(T - mn)
    z = (T - mn - correction) / se
    prob = 2. * distributions.norm.sf(abs(z))

    if r_plus > r_minus:
        return_type = 'r_plus'
        return_val = r_plus
    else:
        return_type = 'r_minus'
        return_val = r_minus

    return WilcoxonResult(T, prob, return_type, return_val)
