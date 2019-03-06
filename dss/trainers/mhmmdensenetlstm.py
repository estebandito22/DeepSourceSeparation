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


class MHMMDenseNetLSTM(SourceSeparator):

    """Class to train and evaluate MDenseNetModel."""

    def __init__(self, n_classes=1, n_shared_layers=3, in_channels=1,
                 hidden_size=128, batch_size=64, lr=0.001,
                 weight_decay=0, num_epochs=100, objective='L2'):
        """
        Initialize MHMMDenseNetLSTM model.

        Args
        ----
            n_classes : int, number of classes.
            n_shared_layers : int, number of shared layes.
            in_channels : int, input channels.
            hidden_size : int, hidden size in lstm layer.
            batch_size : int, batch size for optimization.
            lr : float, learning rate for optimization.
            weight_decay : float, weight decay for optmization.
            num_epochs : int, number of epochs to train for.
            objective : string, 'L1' or 'L2'.

        """
        SourceSeparator.__init__(self)

        # Trainer attributes
        self.n_classes = n_classes
        self.n_shared_layers = n_shared_layers
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.objective = objective

        # Dataset attributes
        self.save_dir = None
        self.pretrained = None

        # Model attributes
        self.model = None
        self.optimizer = None
        self.plateau_scheduler = None
        self.loss_func = None
        self.nn_epoch = 0
        self.best_val_loss = float('inf')

        self.USE_CUDA = torch.cuda.is_available()

    def _init_nn(self, pretrained_state=None):
        """Initialize the nn model for training."""
        self.model = MHMMDenseNetLSTMModel(
            n_classes=self.n_classes, n_shared_layers=self.n_shared_layers,
            in_channels=self.in_channels, hidden_size=self.hidden_size,
            batch_size=self.batch_size)

        if pretrained_state is not None:
            self._load_pretrained(pretrained_state)

        self.optimizer = optim.Adam(
            self.model.parameters(), self.lr, weight_decay=self.weight_decay)

        self.plateau_scheduler = ReduceLROnPlateau(self.optimizer, patience=20)

        if self.objective == 'L1':
            self.loss_func = nn.L1Loss()
        elif self.objective == 'L2':
            self.loss_func = nn.MSELoss()
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
            # batch_size x 512 x 128 x 1
            X = batch_samples['X'].unsqueeze(1)
            # batch_size x nclasses x 1 x 512 x 128
            y = batch_samples['y'].unsqueeze(2)

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
                # batch_size x 512 x 128 x 1
                X = batch_samples['X'].unsqueeze(1)
                # batch_size x nclasses x 1 x 512 x 128
                y = batch_samples['y'].unsqueeze(2)

                if self.USE_CUDA:
                    X = X.cuda()
                    y = y.cuda()

                # detach hidden state
                self.model.detach_hidden(X.size(0))

                # forward pass
                preds, _ = self.model(X)

                # compute loss
                loss = self.loss_func(preds, y)

                bs = X.size(0)
                samples_processed += bs
                val_loss += loss.item() * bs

            val_loss /= samples_processed

        return val_loss

    def fit(self, train_dataset, val_dataset, train_predset, val_predset,
            save_dir, pretrained_state):
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

        """
        # Print settings to output file
        print("Settings:\n\
               Num Classes: {}\n\
               Num Shared Layers: {}\n\
               In Channels: {}\n\
               Hidden Size: {}\n\
               Learning Rate: {}\n\
               Weight Decay: {}\n\
               Objective: {}\n\
               Pretrained: {}\n\
               Save Dir: {}".format(
                   self.n_classes, self.n_shared_layers, self.in_channels,
                   self.hidden_size, self.lr, self.weight_decay,
                   self.objective, bool(pretrained_state),
                   save_dir), flush=True)

        # store datasets
        self.save_dir = save_dir
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
        self._init_nn(pretrained_state)
        train_loss = 0
        train_sdr = None
        train_sir = None
        train_sar = None

        # train loop
        while self.nn_epoch < self.num_epochs + 1:

            train_loaders = self._batch_loaders(train_dataset, k=1)

            for train_loader in train_loaders:
                if self.nn_epoch > 0:
                    print("\nInitializing train epoch...", flush=True)
                    train_loss = self._train_epoch(train_loader)
                    train_sdr, train_sir, train_sar = self.score(
                        train_pred_loader)

                print("\nInitializing val epoch...", flush=True)
                val_loss = self._eval_epoch(val_loader)
                val_sdr, val_sir, val_sar = self.score(val_pred_loader)
                self.plateau_scheduler.step(val_loss)

                # report
                print("\nEpoch: [{}/{}]\n\nTrain Loss Weighted: {}\tTrain SDR: {}\tTrain SIR: {}\tTrain SAR: {}\n\nVal Loss Weighted: {}\tVal SDR: {}\tVal SIR: {}\tVal SAR: {}".format(
                    self.nn_epoch, self.num_epochs, np.round(train_loss, 5),
                    train_sdr, train_sir, train_sar, np.round(val_loss, 5),
                    val_sdr, val_sir, val_sar), flush=True)

                # save best
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
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

        with torch.no_grad():
            for batch_samples in tqdm(loader):

                # prepare training sample
                X = batch_samples['X']
                if X.dim() == 3:
                    full_track = False
                    # batch_size x 1 x 1025 x 129
                    X = X.unsqueeze(1)
                else:
                    bs = X.size(0)
                    ns = X.size(1)
                    full_track = True
                    # batch_size * splits x 1 x 1025 x 129
                    X = X.unsqueeze(2)
                    X = X.view(bs * ns, 1, 1025, 129)

                X_complex = batch_samples['X_complex']
                if X_complex.dim() == 4:
                    # batch_size x 1 x 1025 x 129 x 2
                    X_complex = X_complex.unsqueeze(1)
                else:
                    # batch_size * splits x 1 x 1025 x 129 x 2
                    X_complex = X_complex.unsqueeze(2)
                    X_complex = X_complex.view(bs * ns, 1, 1025, 129, 2)

                # batch_size x nclasses x 1 x 1025 x time samples x 2
                y = batch_samples['y_complex'].unsqueeze(2)
                # batch_size x nclasses
                cs = batch_samples['c']
                # batch_size x 1
                ts = batch_samples['t']

                if self.USE_CUDA:
                    X = X.cuda()
                    X_complex = X_complex.cuda()
                    y = y.cuda()

                # detach hidden state
                self.model.detach_hidden(X.size(0))

                # forward pass
                _, mask = self.model(X)

                if full_track:
                    # batch size x nclasses x 1 x 1025 x time samples
                    mask = mask.view(bs, ns, self.n_classes, 1, 1025, 129)
                    mask = torch.unbind(mask, dim=1)
                    mask = torch.cat(mask, dim=4)
                    # batch_size x 1 x 1025 x time samples x 2
                    X_complex = X_complex.view(bs, ns, 1, 1025, 129, 2)
                    X_complex = torch.unbind(X_complex, dim=1)
                    X_complex = torch.cat(X_complex, dim=3)

                # convert to complex
                # batch size x nclasses x 1 x 1025 x time samples x 2
                X_complex = X_complex.unsqueeze(1).repeat(
                    1, self.n_classes, 1, 1, 1, 1)
                mask = mask.unsqueeze(-1).expand_as(X_complex)
                preds = mask * X_complex
                # batch size x nclasses x 1025 x time samples (complex)
                preds = self._to_complex(preds.squeeze(2))
                ys = self._to_complex(y.squeeze(2))

                all_preds += [preds]
                all_ys += [ys]
                all_cs += [cs]
                all_ts += [ts]

        return all_preds, all_ys, all_cs, all_ts

    def _format_model_subdir(self):
        subdir = "MHMMDenseLSTM_nc{}sl{}ic{}hs{}lr{}wd{}ob{}pt{}".\
                format(self.n_classes, self.n_shared_layers, self.in_channels,
                       self.hidden_size, self.lr, self.weight_decay,
                       self.objective, self.pretrained)

        return subdir
