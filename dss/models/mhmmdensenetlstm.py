"""Class constructing source separation model."""

from copy import deepcopy

import torch
import torch.nn as nn

from dss.models.densenetsublayers import DenseBlock
from dss.models.densenetsublayers import DownSampleBlock
from dss.models.densenetsublayers import DenseLSTMBlock


class MHMMDenseNetLSTMModel(nn.Module):

    """Multi-Head MMDenseNetLSTM source separation model."""

    def __init__(self, n_classes=1, n_shared_layers=3, in_channels=1,
                 hidden_size=512, batch_size=64):
        """Initialize MhMMDenseNetLSTMModel."""
        self.n_classes = n_classes
        self.n_shared_layers = n_shared_layers
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        super(MHMMDenseNetLSTMModel, self).__init__()

        assert self.n_shared_layers <= 3, "shared layers must be <= 3."

        #
        # Convolution Low approx 4.1kHz
        #

        # 1 x 384 x 129
        layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, out_channels=32, kernel_size=(3, 4),
                padding=1),
            # 1 x 384 x 128
            DenseBlock(
                in_channels=32, out_channels=14, kernel_size=3, nlayers=5))

        # 14 x 384 x 128
        layer2 = nn.Sequential(
            DownSampleBlock(channels=14),
            DenseBlock(
                in_channels=14, out_channels=14, kernel_size=3, nlayers=5))

        # 14 x 192 x 64
        layer3 = nn.Sequential(
            DownSampleBlock(channels=14),
            DenseBlock(
                in_channels=14, out_channels=14, kernel_size=3, nlayers=5))

        # 14 x 96 x 32
        layer4 = nn.Sequential(
            DownSampleBlock(channels=14),
            DenseLSTMBlock(
                in_channels=14, out_channels=14, kernel_size=3, nlayers=5,
                in_size=48, batch_size=self.batch_size,
                hidden_size=self.hidden_size))

        # Deconvolution
        # 15 x 48 x 16
        layer5 = nn.ConvTranspose2d(
            in_channels=15, out_channels=15, kernel_size=2, stride=2)

        # 29 x 96 x 32
        layer6 = nn.Sequential(
            DenseBlock(
                in_channels=29, out_channels=14, kernel_size=3, nlayers=5),
            nn.ConvTranspose2d(
                in_channels=14, out_channels=14, kernel_size=2, stride=2))

        # 28 x 192 x 64
        layer7 = nn.Sequential(
            DenseLSTMBlock(
                in_channels=28, out_channels=14, kernel_size=3, nlayers=5,
                in_size=192, batch_size=self.batch_size,
                hidden_size=self.hidden_size),
            nn.ConvTranspose2d(
                in_channels=15, out_channels=15, kernel_size=2, stride=2))

        # 29 x 384 x 128
        layer8 = DenseBlock(
            in_channels=29, out_channels=14, kernel_size=3, nlayers=5)

        # 14 x 384 x 128

        all_layers = [layer1, layer2, layer3, layer4, layer5, layer6,
                      layer7, layer8]

        n = 4
        self.shared_conv_layers_low = nn.ModuleList(
            all_layers[:self.n_shared_layers])
        self.class_conv_layers_low = nn.ModuleList(
            [deepcopy(nn.ModuleList(all_layers[self.n_shared_layers:n]))
             for _ in range(self.n_classes)])
        self.class_deconv_layers_low = nn.ModuleList(
            [deepcopy(nn.ModuleList(all_layers[n:]))
             for _ in range(self.n_classes)])

        #
        # Convolution high 11.025kHz
        #

        # 1 x 641 x 129
        layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, out_channels=32, kernel_size=4,
                padding=1),
            # 1 x 640 x 128
            DenseBlock(
                in_channels=32, out_channels=4, kernel_size=3, nlayers=4))

        # 4 x 640 x 128
        layer2 = nn.Sequential(
            DownSampleBlock(channels=4),
            DenseBlock(
                in_channels=4, out_channels=4, kernel_size=3, nlayers=4))

        # 4 x 320 x 64
        layer3 = nn.Sequential(
            DownSampleBlock(channels=4),
            DenseBlock(
                in_channels=4, out_channels=4, kernel_size=3, nlayers=4))

        # 4 x 160 x 32
        layer4 = nn.Sequential(
            DownSampleBlock(channels=4),
            DenseLSTMBlock(
                in_channels=4, out_channels=4, kernel_size=3, nlayers=4,
                in_size=80, batch_size=self.batch_size,
                hidden_size=self.hidden_size // 4))

        # Deconvolution
        # 5 x 80 x 16
        layer5 = nn.ConvTranspose2d(
            in_channels=5, out_channels=5, kernel_size=2, stride=2)

        # 9 x 160 x 32
        layer6 = nn.Sequential(
            DenseBlock(
                in_channels=9, out_channels=4, kernel_size=3, nlayers=4),
            nn.ConvTranspose2d(
                in_channels=4, out_channels=4, kernel_size=2, stride=2))

        # 8 x 320 x 64
        layer7 = nn.Sequential(
            DenseBlock(
                in_channels=8, out_channels=4, kernel_size=3, nlayers=4),
            nn.ConvTranspose2d(
                in_channels=4, out_channels=4, kernel_size=2, stride=2))

        # 8 x 640 x 128
        layer8 = nn.Sequential(
            DenseBlock(
                in_channels=8, out_channels=4, kernel_size=3, nlayers=4),
            nn.Conv2d(in_channels=4, out_channels=14, kernel_size=1))

        # 14 x 640 x 128

        all_layers = [layer1, layer2, layer3, layer4, layer5, layer6,
                      layer7, layer8]

        n = 4
        self.shared_conv_layers_high = nn.ModuleList(
            all_layers[:self.n_shared_layers])
        self.class_conv_layers_high = nn.ModuleList(
            [deepcopy(nn.ModuleList(all_layers[self.n_shared_layers:n]))
             for _ in range(self.n_classes)])
        self.class_deconv_layers_high = nn.ModuleList(
            [deepcopy(nn.ModuleList(all_layers[n:]))
             for _ in range(self.n_classes)])

        #
        # Convolution Full
        #

        # 1 x 1025 x 129
        layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, out_channels=32, kernel_size=4,
                padding=1),
            # 1 x 1024 x 128
            DenseBlock(
                in_channels=32, out_channels=7, kernel_size=3, nlayers=3))

        # 7 x 1024 x 128
        layer2 = nn.Sequential(
            DownSampleBlock(channels=7),
            DenseBlock(
                in_channels=7, out_channels=7, kernel_size=3, nlayers=3))

        # 7 x 512 x 64
        layer3 = nn.Sequential(
            DownSampleBlock(channels=7),
            DenseBlock(
                in_channels=7, out_channels=7, kernel_size=3, nlayers=4))

        # 7 x 256 x 32
        layer4 = nn.Sequential(
            DownSampleBlock(channels=7),
            DenseLSTMBlock(
                in_channels=7, out_channels=7, kernel_size=3, nlayers=5,
                in_size=128, batch_size=self.batch_size,
                hidden_size=self.hidden_size))

        # 8 x 128 x 16
        layer5 = nn.Sequential(
            DownSampleBlock(channels=8),
            DenseBlock(
                in_channels=8, out_channels=7, kernel_size=3, nlayers=5))

        # Deconvolution
        # 7 x 64 x 8
        layer6 = nn.ConvTranspose2d(
            in_channels=7, out_channels=7, kernel_size=2, stride=2)

        # 15 x 128 x 16
        layer7 = nn.Sequential(
            DenseBlock(
                in_channels=15, out_channels=7, kernel_size=3, nlayers=5),
            nn.ConvTranspose2d(
                in_channels=7, out_channels=7, kernel_size=2, stride=2))

        # 14 x 256 x 32
        layer8 = nn.Sequential(
            DenseBlock(
                in_channels=14, out_channels=7, kernel_size=3, nlayers=4),
            nn.ConvTranspose2d(
                in_channels=7, out_channels=7, kernel_size=2, stride=2))

        # 14 x 512 x 64
        layer9 = nn.Sequential(
            DenseLSTMBlock(
                in_channels=14, out_channels=7, kernel_size=3, nlayers=3,
                in_size=512, batch_size=self.batch_size,
                hidden_size=self.hidden_size),
            nn.ConvTranspose2d(
                in_channels=8, out_channels=8, kernel_size=2, stride=2))

        # 15 x 1024 x 128
        layer10 = DenseBlock(
            in_channels=15, out_channels=7, kernel_size=3, nlayers=3)

        # 7 x 1024 x 128

        all_layers = [layer1, layer2, layer3, layer4, layer5, layer6,
                      layer7, layer8, layer9, layer10]

        n = 5
        self.shared_conv_layers_full = nn.ModuleList(
            all_layers[:self.n_shared_layers])
        self.class_conv_layers_full = nn.ModuleList(
            [deepcopy(nn.ModuleList(all_layers[self.n_shared_layers:n]))
             for _ in range(self.n_classes)])
        self.class_deconv_layers_full = nn.ModuleList(
            [deepcopy(nn.ModuleList(all_layers[n:]))
             for _ in range(self.n_classes)])

        #
        # Final Dense
        #

        # 21 x 1024 x 128
        final_layer = nn.Sequential(
            DenseBlock(
                in_channels=21, out_channels=4, kernel_size=3, nlayers=2),
            nn.Conv2d(
                in_channels=4, out_channels=1, kernel_size=2, padding=1))

        self.class_final_layers = nn.ModuleList(
            [deepcopy(final_layer) for _ in range(self.n_classes)])

        # 1 x 1025 x 129
        self.sigmoid = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        for class_layers in self.class_final_layers:
            nn.init.constant_(class_layers[1].bias, 1)

    def detach_hidden(self, bs):
        """Detach hidden state of LSTMs."""
        # detach conv layer lstms
        for class_layers in self.class_conv_layers_low:
            for layer in class_layers:
                for block in layer:
                    if isinstance(block, DenseLSTMBlock):
                        block.lstm.detach_hidden(bs)
        for class_layers in self.class_conv_layers_high:
            for layer in class_layers:
                for block in layer:
                    if isinstance(block, DenseLSTMBlock):
                        block.lstm.detach_hidden(bs)
        for class_layers in self.class_conv_layers_full:
            for layer in class_layers:
                for block in layer:
                    if isinstance(block, DenseLSTMBlock):
                        block.lstm.detach_hidden(bs)

        # detach deconv layer lstms
        for class_layers in self.class_deconv_layers_low:
            class_layers[2][0].lstm.detach_hidden(bs)
        for class_layers in self.class_deconv_layers_full:
            class_layers[3][0].lstm.detach_hidden(bs)

    @staticmethod
    def _shared_convolutions(x, shared_conv_layers):
        # Shared Convolution
        shared_conv_outs = []
        for layer in shared_conv_layers:
            x = layer(x)
            shared_conv_outs += [x]

        return x, shared_conv_outs

    @staticmethod
    def _class_convolutions(x, class_conv_layers):
        # Class Convolutions
        all_class_conv_outs = []
        for layer_list in class_conv_layers:
            # forward pass and save class specific outputs
            x1 = x
            class_conv_outs = []
            for layer in layer_list:
                x1 = layer(x1)
                class_conv_outs += [x1]
            all_class_conv_outs += [class_conv_outs]

        return all_class_conv_outs

    @staticmethod
    def _class_deconvolutions(shared_conv_outs, all_class_conv_outs,
                              class_deconv_layers):
        # Class Deconvolutions
        x3 = []
        for i, layer_list in enumerate(class_deconv_layers):
            # combine shared and class specific outputs
            conv_outs = shared_conv_outs + all_class_conv_outs[i]
            n_conv_layers = len(conv_outs)
            # first deconv layer has no skip connection
            x2 = layer_list[0](conv_outs[-1])
            # forward pass on remaining deconv layers
            for j, layer in enumerate(layer_list[1:]):
                reverse_layer_number = n_conv_layers - (j + 2)
                sc = conv_outs[reverse_layer_number]
                x2 = layer(torch.cat([x2, sc], dim=1))
            x3 += [x2]
        # list : nclasses, batch size x 1 x height x width
        return x3

    def _class_final_dense(self, x):
        outs = []
        for i, layer in enumerate(self.class_final_layers):
            outs += [layer(x[i])]
        # batch_size x nclasses x 1 x height x width
        return torch.stack(outs, dim=1)

    def forward(self, x):
        """Forward Pass."""
        # low band
        x1, shared_conv_outs = self._shared_convolutions(
            x[:, :, :384, :], self.shared_conv_layers_low)
        all_class_conv_outs = self._class_convolutions(
            x1, self.class_conv_layers_low)
        out_low = self._class_deconvolutions(
            shared_conv_outs, all_class_conv_outs,
            self.class_deconv_layers_low)

        # high band
        x1, shared_conv_outs = self._shared_convolutions(
            x[:, :, 384:, :], self.shared_conv_layers_high)
        all_class_conv_outs = self._class_convolutions(
            x1, self.class_conv_layers_high)
        out_high = self._class_deconvolutions(
            shared_conv_outs, all_class_conv_outs,
            self.class_deconv_layers_high)

        # full band
        x1, shared_conv_outs = self._shared_convolutions(
            x, self.shared_conv_layers_full)
        all_class_conv_outs = self._class_convolutions(
            x1, self.class_conv_layers_full)
        out_full = self._class_deconvolutions(
            shared_conv_outs, all_class_conv_outs,
            self.class_deconv_layers_full)

        out_highlow = [torch.cat([low, high], dim=2)
                       for low, high in zip(out_low, out_high)]
        out_all = [torch.cat([full, highlow], dim=1)
                   for full, highlow in zip(out_full, out_highlow)]

        out = self._class_final_dense(out_all)
        mask = self.sigmoid(out)

        return x.unsqueeze(1).expand_as(mask) * mask, mask
