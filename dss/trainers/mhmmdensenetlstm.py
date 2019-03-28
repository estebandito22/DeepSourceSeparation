"""Classes to train Deep Source Separation Models."""

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dss.trainers.sourceseparator import SourceSeparator
from dss.models.mhmmdensenetlstm import MHMMDenseNetLSTMModel
from dss.models.mhmmdensenetlstm44 import MHMMDenseNetLSTMModel \
    as MHMMDenseNetLSTMModel44


class MHMMDenseNetLSTM(SourceSeparator):

    """Class to train and evaluate MDenseNetModel."""

    def __init__(self, n_classes=1, n_shared_layers=3, in_channels=1,
                 kernel_size=3, hidden_size=128, loss_alphas=False,
                 normalize_masks=False, batch_size=64, lr=0.001,
                 weight_decay=0, num_epochs=100, objective='L1',
                 eval_version='v3', train_class=-1, n_fft=1025, mwf=0):
        """
        Initialize MHMMDenseNetLSTM model.

        Args
        ----
            n_classes : int, number of classes.
            n_shared_layers : int, number of shared layes.
            in_channels : int, input channels.
            kernel_size : int, kernel size in conv layers.
            hidden_size : int, hidden size in lstm layer.
            loss_alphas : bool, use a weighted loss function.
            normalize_masks : bool, normalize predicted masks.
            batch_size : int, batch size for optimization.
            lr : float, learning rate for optimization.
            weight_decay : float, weight decay for optmization.
            num_epochs : int, number of epochs to train for.
            objective : string, 'L1' or 'L2'.
            train_class : int, corresponding class to learn. -1 is all classes.
            n_fft : int, number of fft frequencies.
            mwf : int, kernel size for multichannel wiener filter

        """
        SourceSeparator.__init__(self)

        # Trainer attributes
        self.n_classes = n_classes
        self.n_shared_layers = n_shared_layers
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.hidden_size = hidden_size
        self.loss_alphas = loss_alphas
        self.normalize_masks = normalize_masks
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.objective = objective
        self.eval_version = eval_version
        self.train_class = train_class
        self.n_fft = n_fft
        self.mwf = mwf
        self.n_frames = None
        self.upper_bound_slope = None
        self.lower_bound_slope = None
        self.mag_func = None
        self.threshold = None
        self.chan_swap = None

        # Dataset attributes
        self.save_dir = None
        self.pretrained = None

        # Model attributes
        self.model = None
        self.optimizer = None
        self.plateau_scheduler = None
        self.loss_func = None
        self.nn_epoch = 0
        self.best_val_loss = -float('inf')

        self.USE_CUDA = torch.cuda.is_available()

    def _init_nn(self, pretrained_state=None):
        """Initialize the nn model for training."""
        if self.n_fft == 1025:
            model = MHMMDenseNetLSTMModel
        elif self.n_fft == 2049:
            model = MHMMDenseNetLSTMModel44
        self.model = model(
            n_classes=self.n_classes, n_shared_layers=self.n_shared_layers,
            in_channels=self.in_channels, kernel_size=self.kernel_size,
            hidden_size=self.hidden_size, batch_size=self.batch_size,
            normalize_masks=self.normalize_masks)

        if pretrained_state is not None:
            self._load_pretrained(pretrained_state)

        self.optimizer = optim.Adam(
            self.model.parameters(), self.lr, weight_decay=self.weight_decay)

        self.plateau_scheduler = ReduceLROnPlateau(self.optimizer, patience=50)

        if self.loss_alphas or self.train_class != -1:
            reduction = 'none'
        else:
            reduction = 'mean'

        if self.objective == 'L1':
            self.loss_func = nn.L1Loss(reduction=reduction)
        elif self.objective == 'L2':
            self.loss_func = nn.MSELoss(reduction=reduction)
        else:
            raise ValueError("objective must be L1 or L2.")

        if self.USE_CUDA:
            self.model = self.model.cuda()

    def _train_epoch(self, loader):
        """Train epoch."""
        self.model.train()
        train_loss = 0
        samples_processed = 0

        for batch_samples in tqdm(loader):

            # prepare training sample
            # batch_size x in_channels x 1025 x 129
            X = batch_samples['X']
            # batch_size x nclasses x in_channels x 1025 x 129
            y = batch_samples['y']

            if self.USE_CUDA:
                X = X.cuda()
                y = y.cuda()

            # forward pass
            self.model.zero_grad()

            # detach hidden state
            self.model.detach_hidden(X.size(0))

            preds, _ = self.model(X)

            # backward pass
            loss = self.loss_func(preds, y)
            loss = self._apply_class_mask(loss)
            if self.loss_alphas:
                loss = self._apply_loss_weights(loss, y)
            loss.backward()
            self.optimizer.step()

            # compute train loss
            bs = X.size(0)
            samples_processed += bs
            train_loss += loss.item() * bs

        train_loss /= samples_processed

        return train_loss

    def _eval_epoch(self, loader):
        """Eval epoch."""
        self.model.eval()
        val_loss = 0
        samples_processed = 0

        with torch.no_grad():
            for batch_samples in tqdm(loader):

                # prepare training sample
                # batch_size x in_channels x 1025 x 129
                X = batch_samples['X']
                # batch_size x nclasses x in_channels x 1025 x 129
                y = batch_samples['y']

                if self.USE_CUDA:
                    X = X.cuda()
                    y = y.cuda()

                # detach hidden state
                self.model.detach_hidden(X.size(0))

                # forward pass
                preds, _ = self.model(X)

                # compute loss
                loss = self.loss_func(preds, y)
                loss = self._apply_class_mask(loss)
                if self.loss_alphas:
                    loss = self._apply_loss_weights(loss, y)

                bs = X.size(0)
                samples_processed += bs
                val_loss += loss.item() * bs

            val_loss /= samples_processed

        return val_loss

    def fit(self, train_dataset, val_dataset, train_predset, val_predset,
            save_dir, pretrained_state, warm_start):
        """
        Train the NN model.

        Args
        ----
            train_dataset : PyTorch dataset, training data.
            val_dataset : PyTorch dataset, validation data.
            train_predset : PyTorch dataset, predicton training data.
            val_predset : PyTorch dataset, prediction validation data.
            save_dir: directory to save nn_model.
            pretrained_state : dict, pretrained model state dict.
            warm_start : bool, continue from previous results.

        """
        # Print settings to output file
        print("Settings:\n\
               Num Classes: {}\n\
               Num Shared Layers: {}\n\
               In Channels: {}\n\
               Kernel Size: {}\n\
               Hidden Size: {}\n\
               Loss Alphas: {}\n\
               Normalize Masks: {}\n\
               Eval Version: {}\n\
               Mag Func: {}\n\
               N Frames: {}\n\
               N FFT: {}\n\
               Curriculum Upper Bound Slope: {}\n\
               Curriculum Lower Bound Slope: {}\n\
               Threshold: {}\n\
               Channel Swap: {}\n\
               Learning Rate: {}\n\
               Weight Decay: {}\n\
               Objective: {}\n\
               Pretrained: {}\n\
               Train Class: {}\n\
               Save Dir: {}".format(
                   self.n_classes, self.n_shared_layers, self.in_channels,
                   self.kernel_size, self.hidden_size,
                   self.loss_alphas, self.normalize_masks, self.eval_version,
                   train_dataset.mag_func, train_dataset.n_frames,
                   self.n_fft, train_dataset.upper_bound_slope,
                   train_dataset.lower_bound_slope, train_dataset.threshold,
                   train_dataset.chan_swap, self.lr, self.weight_decay,
                   self.objective, bool(pretrained_state), self.train_class,
                   save_dir), flush=True)

        # store datasets
        self.save_dir = save_dir
        self.n_frames = train_dataset.n_frames
        self.upper_bound_slope = train_dataset.upper_bound_slope
        self.lower_bound_slope = train_dataset.lower_bound_slope
        self.mag_func = train_dataset.mag_func
        self.threshold = train_dataset.threshold
        self.chan_swap = train_dataset.chan_swap
        self.pretrained = bool(pretrained_state)

        # initialize constant loaders
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=8)

        val_pred_loader = DataLoader(
            val_predset, batch_size=self.batch_size, shuffle=False,
            num_workers=8)

        train_pred_loader = DataLoader(
            train_predset, batch_size=self.batch_size, shuffle=False,
            num_workers=8)

        # initialize neural network and training variables
        if not warm_start:
            self._init_nn(pretrained_state)

        train_loss = 0
        train_sdr = None
        train_sir = None
        train_sar = None
        train_avg_sdr = None

        # train loop
        while self.nn_epoch < self.num_epochs + 1:

            train_loaders = self._batch_loaders(train_dataset, k=1)

            for train_loader in train_loaders:

                train_loader.dataset.dataset.set_concentration(self.nn_epoch)

                if self.nn_epoch > 0:
                    print("\nInitializing train epoch...", flush=True)
                    train_loss = self._train_epoch(train_loader)
                    # train_sdr, train_sir, train_sar = self.score(
                    #     train_pred_loader)

                print("\nInitializing val epoch...", flush=True)
                val_loss = self._eval_epoch(val_loader)
                val_sdr, val_sir, val_sar = self.score(val_pred_loader)

                # report
                if train_sdr is not None:
                    train_avg_sdr = np.round(
                        np.mean(list(train_sdr.values())), 5)
                val_avg_sdr = np.round(np.mean(list(val_sdr.values())), 5)
                self._report(
                    {'L1': np.round(train_loss, 5), 'Avg SDR': train_avg_sdr},
                    train_sdr, train_sir, train_sar,
                    {'L1': np.round(val_loss, 5), 'Avg SDR': val_avg_sdr},
                    val_sdr, val_sir, val_sar)

                self.plateau_scheduler.step(-val_avg_sdr)

                # save best
                if val_avg_sdr > self.best_val_loss:
                    self.best_val_loss = val_avg_sdr
                    self.save()
                self.nn_epoch += 1

    def predict(self, loader):
        """
        Predict for an input.

        Args
        ----
            loader : PyTorch DataLoader.

        """
        self.model.eval()
        all_preds = []
        all_ys = []
        all_cs = []
        all_ts = []
        all_ms = []

        if isinstance(loader.dataset, torch.utils.data.Subset):
            n_frames = loader.dataset.dataset.n_frames
        else:
            n_frames = loader.dataset.n_frames

        with torch.no_grad():
            for batch_samples in tqdm(loader):

                # prepare training sample
                X = batch_samples['X']
                if X.dim() == 4:
                    full_track = False
                    # batch_size x in_channels x 1025 x 129
                else:
                    bs = X.size(0)
                    ns = X.size(1)
                    full_track = True
                    # batch_size * splits x in_channels x 1025 x 129
                    X = X.view(bs * ns, self.in_channels, self.n_fft, n_frames)

                # batch_size x in_channels x 1025 x 129 x 2
                X_complex = batch_samples['X_complex']
                if X_complex.dim() != 5:
                    # batch_size * splits x in_channels x 1025 x 129 x 2
                    X_complex = X_complex.view(
                        bs * ns, self.in_channels, self.n_fft, n_frames, 2)

                # batch_size x nclasses x in_channels x 1025 x time samples x 2
                y = batch_samples['y_complex']
                # batch_size x nclasses
                cs = batch_samples['c']
                # batch_size x 1
                ts = batch_samples['t']

                if self.USE_CUDA:
                    X = X.cuda()
                    X_complex = X_complex.cuda()
                    y = y.cuda()

                if X.size(0) > 8:
                    X_list = torch.split(X, 8, dim=0)
                else:
                    X_list = [X]

                masks_list = []
                for X in X_list:
                    # detach hidden state
                    self.model.detach_hidden(X.size(0))
                    # forward pass
                    _, mask = self.model(X)
                    masks_list += [mask]
                mask = torch.cat(masks_list, dim=0)

                if full_track:
                    # batch size x nclasses x in_channels x 1025 x time samples
                    mask = mask.view(
                        bs, ns, self.n_classes, self.in_channels, self.n_fft,
                        n_frames)
                    mask = torch.unbind(mask, dim=1)
                    mask = torch.cat(mask, dim=4)
                    # batch_size x in_channels x 1025 x time samples x 2
                    X_complex = X_complex.view(
                        bs, ns, self.in_channels, self.n_fft, n_frames, 2)
                    X_complex = torch.unbind(X_complex, dim=1)
                    X_complex = torch.cat(X_complex, dim=3)

                # convert to complex
                # batch size x nclasses x in_channels x 1025 x time samples x 2
                X_complex = X_complex.unsqueeze(1).repeat(
                    1, self.n_classes, 1, 1, 1, 1)
                X_complex = self._to_complex(X_complex)
                preds = mask.cpu().numpy() * X_complex
                # batch size x nclasses x in_channels x 1025 x time samples
                ys = self._to_complex(y)

                all_preds += [preds]
                all_ys += [ys]
                all_cs += [cs]
                all_ts += [ts]
                all_ms += [mask.cpu().numpy()]

        return all_preds, all_ys, all_cs, all_ts, all_ms

    def _apply_class_mask(self, loss):
        if self.train_class == -1:
            return loss
        if self.train_class == -999 and not self.model.training:
            return torch.mean(loss)
        if self.train_class == -999:
            train_class = np.random.randint(0, self.n_classes)
        else:
            train_class = self.train_class
        mask = torch.zeros_like(loss)
        mask[:, train_class, :, :, :] += 1
        return torch.mean(loss * mask)

    def _format_model_subdir(self):
        subdir = "MHMMDenseLSTM_nc{}sl{}ic{}hs{}lr{}wd{}ob{}pt{}la{}ev{}mf{}nf{}tc{}ks{}us{}ls{}nm{}th{}nfft{}cs{}".\
                format(self.n_classes, self.n_shared_layers, self.in_channels,
                       self.hidden_size, self.lr, self.weight_decay,
                       self.objective, self.pretrained,
                       self.loss_alphas, self.eval_version, self.mag_func,
                       self.n_frames, self.train_class, self.kernel_size,
                       self.upper_bound_slope, self.lower_bound_slope,
                       self.normalize_masks, self.threshold, self.n_fft,
                       self.chan_swap)

        return subdir
