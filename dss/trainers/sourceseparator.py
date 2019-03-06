"""Classes to train Deep Source Separation Models."""

import os
from copy import deepcopy
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from librosa.core import istft
from mir_eval.separation import bss_eval_sources

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

from dss.trainers.trainer import Trainer


class SourceSeparator(Trainer):

    """Class to train and evaluate source separation models."""

    def __init__(self):
        """Initialize SourceSeparator Trainer."""
        Trainer.__init__(self)

        # Trainer attributes
        self.batch_size = None

        # Model attributes
        self.model = None
        self.nn_epoch = 0

        self.save_dir = None
        self.USE_CUDA = torch.cuda.is_available()

    def fit(self):
        """Fit function."""
        raise NotImplementedError("Not yet implemented!")

    def score(self, loader):
        """
        Score the model.

        Args
        ----
          loader : PyTorch DataLoader.

        """
        self.model.eval()
        class_sdr = defaultdict(list)
        class_sir = defaultdict(list)
        class_sar = defaultdict(list)

        # list of batches
        preds, ys, cs, ts = self.predict(loader)

        # for each batch
        for b_preds, b_ys, b_cs, b_ts in tqdm(list(zip(preds, ys, cs, ts))):
            # for each sample
            for pred, y, c, t in zip(b_preds, b_ys, b_cs, b_ts):
                pred_recons = []
                y_recons = []
                pred_cs = []
                # for each class
                for i, (c_pred, c_y, c_c) in enumerate(zip(pred, y, c)):
                    # if the class exists in the source signal
                    if c_c == 1 and np.abs(c_y).sum() > 0:
                        c_pred = c_pred[:, :t]
                        c_y = c_y[:, :t]
                        pred_recons += [istft(
                            c_pred, hop_length=512, win_length=2048)]
                        y_recons += [istft(
                            c_y, hop_length=512, win_length=2048)]
                        pred_cs += [i]
                # possible to sample from targets that are all zeros
                if pred_recons:
                    pred_recons = np.stack(pred_recons)
                    # possible to predict all zeros...
                    # TODO: Figure out how to handle this case properly
                    if np.abs(pred_recons.sum()) > 0:
                        y_recons = np.stack(y_recons)
                        # nclassex x time
                        sdr, sir, sar, _ = bss_eval_sources(
                            y_recons, pred_recons, compute_permutation=False)
                        for m1, m2, m3, cl in zip(sdr, sir, sar, pred_cs):
                            class_sdr[cl] += [m1]
                            class_sir[cl] += [m2]
                            class_sar[cl] += [m3]

        for k, v in class_sdr.items():
            class_sdr[k] = np.round(np.mean(v), 2)
        for k, v in class_sir.items():
            class_sir[k] = np.round(np.mean(v), 2)
        for k, v in class_sar.items():
            class_sar[k] = np.round(np.mean(v), 2)

        return class_sdr, class_sir, class_sar

    def predict(self, loader):
        """
        Predict for an input.

        Args
        ----
            loader : PyTorch DataLoader.

        """
        self.model.eval()
        raise NotImplementedError("Not yet implemented!")

    def _batch_loaders(self, dataset, k=None):
        batches = dataset.get_batches(k)
        loaders = []
        for subset_batch_indexes in batches:
            subset = Subset(dataset, subset_batch_indexes)
            loader = DataLoader(
                subset, batch_size=self.batch_size, shuffle=True,
                num_workers=8)
            loaders += [loader]
        return loaders

    @staticmethod
    def _to_complex(x):
        x = x.cpu().numpy()
        return x[..., 0] + 1j * x[..., 1]

    def _load_pretrained(self, pretrained_state):
        new_state = deepcopy(self.model.state_dict())
        for k in new_state.keys():
            if k.find('shared') > -1:
                new_state[k].copy_(pretrained_state[k])
            else:
                pre_key = k.split('.')
                pre_key[1] = str(0)
                pre_key = '.'.join(pre_key)
                new_state[k].copy_(pretrained_state[pre_key])
        self.model.load_state_dict(new_state)
        print("Loaded pretrained model...")

    def save(self):
        """
        Save model.

        Args
        ----
            models_dir: path to directory for saving NN models.

        """
        if (self.model is not None) and (self.save_dir is not None):

            model_dir = self._format_model_subdir()

            if not os.path.isdir(os.path.join(self.save_dir, model_dir)):
                os.makedirs(os.path.join(self.save_dir, model_dir))

            filename = "epoch_{}".format(self.nn_epoch) + '.pth'
            fileloc = os.path.join(self.save_dir, model_dir, filename)
            with open(fileloc, 'wb') as file:
                torch.save({'state_dict': self.model.state_dict(),
                            'trainer_dict': self.__dict__}, file)

    def load(self, model_dir, epoch):
        """
        Load a previously trained model.

        Args
        ----
            model_dir : directory where models are saved.
            epoch : epoch of model to load.
            
        """
        epoch_file = "epoch_{}".format(epoch) + '.pth'
        model_file = os.path.join(model_dir, epoch_file)
        with open(model_file, 'rb') as model_dict:
            if torch.cuda.is_available():
                checkpoint = torch.load(model_dict)
            else:
                checkpoint = torch.load(model_dict, map_location='cpu')

        for (k, v) in checkpoint['trainer_dict'].items():
            setattr(self, k, v)

        self.USE_CUDA = torch.cuda.is_available()
        self._init_nn()
        self.model.load_state_dict(checkpoint['state_dict'])
