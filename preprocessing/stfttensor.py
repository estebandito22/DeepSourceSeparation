"""Class to transform .mp3 to log-melspectrogram."""

import os
import warnings
import torch
import pandas as pd
import numpy as np
import librosa


class STFTTransformer(object):

    """Class to transform audio to complex spectrogram tensor."""

    def __init__(self, n_fft=1024, hop_length=512, mono=True, sr=22050):
        """Initialize transformer."""
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.mono = mono
        self.sr = sr

    @staticmethod
    def librosa_to_tensor(stft):
        """Convert a librosa complex stft to a pytorch tensor."""
        real = np.real(stft)
        imag = np.imag(stft)
        stft = np.stack([real, imag], axis=2)
        return torch.from_numpy(stft)

    def transform(self, meta_file, save_dir, overwrite=False,
                  groups=4, group=0, out_q=None):
        """Transform mp3s to logmelspectrograms."""
        url_ids = []
        stft_paths = []
        track_ids = []
        song_ids = []
        instruments = []
        volumes = []
        splits = []
        metadata = pd.read_csv(meta_file)
        files = metadata['file_path']
        trackId = metadata['trackId']
        songId = metadata['songId']
        instrument = metadata['instrument']
        volume = metadata['trackVolume']
        if 'split' not in metadata.columns:
            metadata['split'] = None
        split = metadata['split']

        k = int(np.ceil(len(files) / groups))
        start = k * group
        end = k * (group+1)
        files = files[start:end]
        trackId = trackId[start:end]
        songId = songId[start:end]
        instrument = instrument[start:end]
        volume = volume[start:end]
        split = split[start:end]

        print("save dir: {}".format(save_dir), flush=True)

        for element in zip(files, trackId, songId, instrument, volume, split):
            file, track_id, song_id, inst, vol, splt = element
            url_id = file.split("/")[-1].split(".")[0]
            f = os.path.join(save_dir, track_id + "_stft.txt")

            if not os.path.isfile(f) or overwrite:

                try:
                    audio, _ = librosa.load(file, sr=self.sr, mono=self.mono)
                    if self.mono:
                        stft = librosa.core.stft(
                            y=audio, n_fft=self.n_fft,
                            hop_length=self.hop_length)
                        stft = self.librosa_to_tensor(stft)
                    else:
                        stft0 = librosa.core.stft(
                            y=audio[0, :], n_fft=self.n_fft,
                            hop_length=self.hop_length)
                        stft1 = librosa.core.stft(
                            y=audio[1, :], n_fft=self.n_fft,
                            hop_length=self.hop_length)
                        stft0 = self.librosa_to_tensor(stft0)
                        stft1 = self.librosa_to_tensor(stft1)
                        stft = torch.stack([stft0, stft1])
                    torch.save(stft, f)
                    print("Saving to {}".format(f), flush=True)
                except:
                    warnings.warn(
                        "Could not load file: {}".format(file))

            if os.path.isfile(f):
                url_ids += [url_id]
                stft_paths += [f]
                track_ids += [track_id]
                song_ids += [song_id]
                instruments += [inst]
                volumes += [vol]
                splits += [splt]

        return_dict = {'stft_path': stft_paths, 'trackId': track_ids,
                       'songId': song_ids, 'instrument': instruments,
                       'trackVolume': volumes, 'urlId': url_ids,
                       'split': splits}
        out_q.put(return_dict)
