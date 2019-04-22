"""Classes to train Deep Source Separation Models."""

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dss.trainers.sourceseparator import SourceSeparator
from dss.models.pretrainmhmmdensenetlstm import PretrainMHMMDenseNetLSTMModel


class PretrainMHMMDenseNetLSTM(SourceSeparator):

    """Class to pretrain and evaluate MhMMDenseNetLSTM model."""

    def __init__(self, n_classes=1, n_shared_layers=3, in_channels=1,
                 hidden_size=128, batch_size=64, lr=0.001, weight_decay=0,
                 num_epochs=100, objective='L1'):
        """
        Initialize Pretrain MHMMDenseNetLSTM model.

        Args
        ----
            n_classes
            negative_slope : float, slope in leaky relu.
            dropout : float, dropout rate.
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

        # Model attributes
        self.model = None
        self.optimizer = None
        self.loss_func = None
        self.nn_epoch = 0
        self.best_val_loss = float('inf')

        self.USE_CUDA = torch.cuda.is_available()

    def _init_nn(self):
        """Initialize the nn model for training."""
        self.model = PretrainMHMMDenseNetLSTMModel(
            n_classes=self.n_classes, n_shared_layers=self.n_shared_layers,
            in_channels=self.in_channels, hidden_size=self.hidden_size,
            batch_size=self.batch_size)

        self.optimizer = optim.Adam(
            self.model.parameters(), self.lr, weight_decay=self.weight_decay)

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
            # batch_size x 1 x 1025 x 129
            X = batch_samples['X'].unsqueeze(1)

            if self.USE_CUDA:
                X = X.cuda()

            # forward pass
            self.model.zero_grad()

            # detach hidden state
            self.model.detach_hidden(X.size(0))

            preds = self.model(X)

            # backward pass
            bs = X.size(0)
            loss = self.loss_func(preds.view(bs, -1), X.view(bs, -1))

            loss.backward()
            self.optimizer.step()

            # compute train loss
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

                if self.USE_CUDA:
                    X = X.cuda()

                # detach hidden state
                self.model.detach_hidden(X.size(0))

                # forward pass
                preds = self.model(X)

                # compute loss
                bs = X.size(0)
                loss = self.loss_func(preds.view(bs, -1), X.view(bs, -1))

                samples_processed += bs
                val_loss += loss.item() * bs

            val_loss /= samples_processed

        return val_loss

    def fit(self, train_dataset, val_dataset, save_dir):
        """
        Train the NN model.

        Args
        ----
            train_dataset : PyTorch dataset, training data.
            val_dataset : PyTorch dataset, validation data.
            save_dir: directory to save nn_model

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
               Save Dir: {}".format(
                   self.n_classes, self.n_shared_layers, self.in_channels,
                   self.hidden_size, self.lr, self.weight_decay,
                   self.objective, save_dir), flush=True)

        # store datasets
        self.save_dir = save_dir

        # initialize constant loaders
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=8)

        # initialize neural network and training variables
        self._init_nn()
        train_loss = 0

        # train loop
        while self.nn_epoch < self.num_epochs + 1:

            train_loaders = self._batch_loaders(train_dataset, k=2)

            for train_loader in train_loaders:
                if self.nn_epoch > 0:
                    print("\nInitializing train epoch...", flush=True)
                    train_loss = self._train_epoch(train_loader)

                print("\nInitializing val epoch...", flush=True)
                val_loss = self._eval_epoch(val_loader)

                # report
                print("\nEpoch: [{}/{}]\tTrain Loss: {}\tVal Loss: {}".format(
                    self.nn_epoch, self.num_epochs, train_loss,
                    val_loss), flush=True)

                # save best
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save()
                self.nn_epoch += 1

    def _format_model_subdir(self):
        subdir = "PRE_MHMMDenseNetLSTSM_nc{}sl{}ic{}hs{}lr{}wd{}ob{}".\
                format(self.n_classes, self.n_shared_layers, self.in_channels,
                       self.hidden_size, self.lr, self.weight_decay,
                       self.objective)
        return subdir
