"""Classes to train Deep Source Separation Models."""

import os
import os.path as op
from copy import deepcopy
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from librosa.core import istft
from librosa.core import resample
from mir_eval.separation import bss_eval_sources
# from museval.metrics import bss_eval
import musdb
from museval import evaluate
from museval import EvalStore

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

from dss.trainers.trainer import Trainer


class SourceSeparator(Trainer):

    """Class to train and evaluate source separation models."""

    def __init__(self):
        """Initialize SourceSeparator Trainer."""
        Trainer.__init__(self)

    def fit(self):
        """Fit function."""
        raise NotImplementedError("Not yet implemented!")

    def score(self, loader, framewise=False, save_dir=None):
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

        # only perform framewise evaluation at testing time
        if self.n_fft == 1025:
            rate = 22050
            hop = 512
            win = 2048
        elif self.n_fft == 2049:
            rate = 44100
            hop = 1024
            win = 4096
        if not framewise:
            rate = np.inf

        if save_dir:
            class_map = {0: 'bass', 1: 'drums', 2: 'other', 3: 'vocals'}
            mus = musdb.DB(root_dir="data/musdb18")

        # list of batches
        preds, ys, cs, ts, _, nm = self.predict(loader)

        # for each batch
        for b_preds, b_ys, b_cs, b_ts, b_nm in tqdm(list(zip(preds, ys, cs, ts, nm))):
            # for each sample
            for pred, y, c, t, n in zip(b_preds, b_ys, b_cs, b_ts, b_nm):
                pred_recons = []
                y_recons = []
                pred_cs = []
                pred_recons_dict = defaultdict(list)
                y_recons_dict = defaultdict(list)
                # for each class
                for i, (c_pred, c_y, c_c) in enumerate(zip(pred, y, c)):
                    # if the class exists in the source signal
                    if c_c == 1 and np.abs(c_y).sum() > 0:
                        c_pred = c_pred[..., :t]
                        c_y = c_y[..., :t]
                        # predictions can be over multiple channels
                        pred_recon = []
                        y_recon = []
                        for c_pred_chan, c_y_chan in zip(c_pred, c_y):
                            pred_recon += [istft(
                                c_pred_chan, hop_length=hop, win_length=win)]
                            y_recon += [istft(
                                c_y_chan, hop_length=hop, win_length=win)]
                        pred_recon = np.stack(pred_recon, axis=-1)
                        y_recon = np.stack(y_recon, axis=-1)
                        # accumulate list of reconstructions for stacking
                        pred_recons += [pred_recon]
                        y_recons += [y_recon]
                        pred_cs += [i]
                        if save_dir:
                            pred_recons_dict[class_map[i]] = pred_recon
                            y_recons_dict[class_map[i]] = y_recon
                # possible to sample from targets that are all zeros
                if pred_recons:
                    pred_recons = np.stack(pred_recons)
                    # possible to predict all zeros...
                    # TODO: Figure out how to handle this case properly
                    if np.abs(pred_recons.sum()) > 0:
                        y_recons = np.stack(y_recons)
                        # nclassex x time
                        if self.eval_version == 'v3':
                            sdr, sir, sar, _ = bss_eval_sources(
                                y_recons, pred_recons,
                                compute_permutation=False)
                        elif self.eval_version == 'v4':
                            if save_dir:
                                name = loader.dataset.metadata.at[
                                    int(n.cpu().numpy()), 'urlId']
                                track = mus.load_mus_tracks(
                                    tracknames=[name])[0]
                                sdr, isr, sir, sar = evaluate(
                                    y_recons, pred_recons, win=rate, hop=rate,
                                    padding=True)
                                data = self._to_evalstore(
                                    sdr, sir, isr, sar, rate, rate, class_map)
                                self._save_framewise(data, save_dir, track)
                                continue
                            else:
                                sdr, isr, sir, sar = evaluate(
                                    y_recons, pred_recons, win=rate, hop=rate,
                                    padding=True)
                                cmb_sdr = np.concatenate([x for x in sdr])
                                sdr = np.nanmean(sdr, axis=1)
                                sir = np.nanmean(sir, axis=1)
                                sar = np.nanmean(sar, axis=1)
                        for m1, m2, m3, cl in zip(sdr, sir, sar, pred_cs):
                            class_sdr[cl] += [m1]
                            class_sir[cl] += [m2]
                            class_sar[cl] += [m3]

        class_sdr_out = defaultdict(list)
        class_sir_out = defaultdict(list)
        class_sar_out = defaultdict(list)

        class_sdr_out['median'] = {k: np.round(np.median(v), 2)
                                   for k, v in class_sdr.items()}
        class_sdr_out['mean'] = {k: np.round(np.mean(v), 2)
                                 for k, v in class_sdr.items()}
        class_sir_out['median'] = {k: np.round(np.median(v), 2)
                                   for k, v in class_sir.items()}
        class_sir_out['mean'] = {k: np.round(np.mean(v), 2)
                                 for k, v in class_sir.items()}
        class_sar_out['median'] = {k: np.round(np.median(v), 2)
                                   for k, v in class_sar.items()}
        class_sar_out['mean'] = {k: np.round(np.mean(v), 2)
                                 for k, v in class_sar.items()}

        return class_sdr_out, class_sir_out, class_sar_out, cmb_sdr

    def _to_evalstore(self, sdr, sir, isr, sar, win, hop, class_map):
        data = EvalStore(win=win, hop=hop)
        # iterate over all evaluation results except for vocals
        for i, target in class_map.items():
            # if target == 'vocals' and has_acc:
            #     continue

            values = {
                "SDR": sdr[i].tolist(),
                "SIR": sir[i].tolist(),
                "ISR": isr[i].tolist(),
                "SAR": sar[i].tolist()
            }

            data.add_target(
                target_name=target,
                values=values
            )
        return data

    def _save_framewise(self, data, output_dir, track):
        # validate against the schema
        data.validate()

        try:
            subset_path = op.join(
                output_dir,
                track.subset
            )

            if not op.exists(subset_path):
                os.makedirs(subset_path)

            with open(
                op.join(subset_path, track.name) + '.json', 'w+'
            ) as f:
                f.write(data.json)

        except (IOError):
            pass

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
                pre_key[1] = pre_key[1][:-1] + str(0)
                pre_key = '.'.join(pre_key)
                new_state[k].copy_(pretrained_state[pre_key])
        self.model.load_state_dict(new_state)
        print("Loaded pretrained model...")

    def _report(self, train_loss, train_sdr, train_sir, train_sar, val_loss,
                val_sdr, val_sir, val_sar):
        e = 'Epoch: [{}/{}]'.format(self.nn_epoch, self.num_epochs)
        tl = 'Train Loss Weighted: {}'.format(train_loss)
        tsdr = 'Train SDR: {}'.format(train_sdr)
        tsir = 'Train SIR: {}'.format(train_sir)
        tsar = 'Train SAR: {}'.format(train_sar)
        vl = 'Val Loss Weighted: {}'.format(val_loss)
        vsdr = 'Val SDR: {}'.format(val_sdr)
        vsir = 'Val SIR: {}'.format(val_sir)
        vsar = 'Val SAR: {}'.format(val_sar)

        text = ["\n", e, "\n\n", tl, "\t", tsdr, "\t", tsir, "\t", tsar,
                "\n\n", vl, "\t", vsdr, "\t", vsir, "\t", vsar]

        print(''.join(text), flush=True)

    def _apply_loss_weights(self, x, y):
        # y_norms = torch.norm(y, dim=(3, 4)).mean(dim=2).mean(dim=0)
        # y_alphas = (y_norms[0] / y_norms) / torch.sum(y_norms[0] / y_norms)
        # if not y_alphas.sum() < float('inf'):
        #     y_alphas = torch.ones((1, self.n_classes)).div_(self.n_classes)
        #     if self.USE_CUDA:
        #         y_alphas = y_alphas.cuda(self.device)
        y_alphas = torch.tensor([[0.232, 0.262, 0.209, 0.297]])
        if self.USE_CUDA:
            y_alphas = y_alphas.cuda(self.device)
        res = (x.transpose_(1, 0).contiguous().view(4, -1).mean(dim=1)
               * y_alphas.squeeze())
        return res.sum()

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
        torch.set_rng_state(self.torch_rng_state)
        np.random.set_state(self.numpy_rng_state)
        self.nn_epoch += 1
