"""Class constructing source separation model."""

from copy import deepcopy

import torch
import torch.nn as nn


class MHUNetModel(nn.Module):

    """Multi-Head UNet source separation model."""

    def __init__(self, n_classes=1, n_shared_layers=3, in_channels=1,
                 negative_slope=0.2, dropout=0.5):
        """Initialize MHUNetModel."""
        self.n_classes = n_classes
        self.n_shared_layers = n_shared_layers
        self.in_channels = in_channels
        self.negative_slope = negative_slope
        self.dropout = dropout
        super(MHUNetModel, self).__init__()

        # Convolution
        # 1025 x 129 x 1
        layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, out_channels=16, kernel_size=5,
                stride=2, padding=1),
            nn.LeakyReLU(negative_slope=self.negative_slope))

        # 512 x 64 x 16
        layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=5, stride=2,
                padding=2),
            nn.BatchNorm2d(num_features=32, track_running_stats=False),
            nn.LeakyReLU(negative_slope=self.negative_slope))

        # 256 x 32 x 32
        layer3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=5, stride=2,
                padding=2),
            nn.BatchNorm2d(num_features=64, track_running_stats=False),
            nn.LeakyReLU(negative_slope=self.negative_slope))

        # 128 x 16 x 64
        layer4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=5, stride=2,
                padding=2),
            nn.BatchNorm2d(num_features=128, track_running_stats=False),
            nn.LeakyReLU(negative_slope=self.negative_slope))

        # 64 x 8 x 128
        layer5 = nn.Sequential(

            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=5, stride=2,
                padding=2),
            nn.BatchNorm2d(num_features=256, track_running_stats=False),
            nn.LeakyReLU(negative_slope=self.negative_slope))

        # 32 x 4 x 256
        layer6 = nn.Sequential(
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=5, stride=2,
                padding=2),
            nn.ReLU())

        # Deconvolution
        # 16 x 2 x 512
        layer7 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=512, out_channels=256, kernel_size=5, stride=2,
                padding=2, output_padding=1),
            nn.BatchNorm2d(num_features=256, track_running_stats=False),
            nn.Dropout(p=self.dropout),
            nn.ReLU())

        # 32 x 4 x 256
        layer8 = nn.Sequential(
            # nn.Dropout(p=self.dropout),
            nn.ConvTranspose2d(
                in_channels=512, out_channels=128, kernel_size=5, stride=2,
                padding=2, output_padding=1),
            nn.BatchNorm2d(num_features=128, track_running_stats=False),
            nn.Dropout(p=self.dropout),
            nn.ReLU())

        # 64 x 8 x 128
        layer9 = nn.Sequential(
            # nn.Dropout(p=self.dropout),
            nn.ConvTranspose2d(
                in_channels=256, out_channels=64, kernel_size=5, stride=2,
                padding=2, output_padding=1),
            nn.BatchNorm2d(num_features=64, track_running_stats=False),
            nn.Dropout(p=self.dropout),
            nn.ReLU())

        # 128 x 16 x 64
        layer10 = nn.Sequential(
            # nn.Dropout(p=self.dropout),
            nn.ConvTranspose2d(
                in_channels=128, out_channels=32, kernel_size=5, stride=2,
                padding=2, output_padding=1),
            nn.BatchNorm2d(num_features=32, track_running_stats=False),
            nn.ReLU())

        # 256 x 32 x 32
        layer11 = nn.Sequential(
            # nn.Dropout(p=self.dropout),
            nn.ConvTranspose2d(
                in_channels=64, out_channels=16, kernel_size=5, stride=2,
                padding=2, output_padding=1),
            nn.BatchNorm2d(num_features=16, track_running_stats=False),
            nn.ReLU())

        # 512 x 64 x 16
        layer12 = nn.ConvTranspose2d(
            in_channels=32, out_channels=1, kernel_size=5, stride=2,
            padding=1)

        # 1025 x 129 x 1
        self.sigmoid = nn.Sigmoid()

        all_layers = [layer1, layer2, layer3, layer4, layer5, layer6, layer7,
                      layer8, layer9, layer10, layer11, layer12]

        self.shared_conv_layers = nn.ModuleList(
            all_layers[:self.n_shared_layers])
        self.class_conv_layers = nn.ModuleList(
            [deepcopy(nn.ModuleList(all_layers[self.n_shared_layers:6]))
             for _ in range(self.n_classes)])
        self.class_deconv_layers = nn.ModuleList(
            [deepcopy(nn.ModuleList(all_layers[6:]))
             for _ in range(self.n_classes)])

        self.n_conv_layers = 6

        self._init_weights()

    def _init_weights(self):
        for class_layers in self.class_deconv_layers:
            nn.init.constant_(class_layers[-1].bias, 1)

    def _shared_convolutions(self, x):
        # Shared Convolution
        shared_conv_outs = []
        for layer in self.shared_conv_layers:
            x = layer(x)
            shared_conv_outs += [x]

        return x, shared_conv_outs

    def _class_convolutions(self, x):
        # Class Convolutions
        all_class_conv_outs = []
        for layer_list in self.class_conv_layers:
            # forward pass and save class specific outputs
            x1 = x
            class_conv_outs = []
            for layer in layer_list:
                x1 = layer(x1)
                class_conv_outs += [x1]
            all_class_conv_outs += [class_conv_outs]

        return all_class_conv_outs

    def _class_deconvolutions(self, shared_conv_outs, all_class_conv_outs):
        # Class Deconvolutions
        x3 = []
        for i, layer_list in enumerate(self.class_deconv_layers):
            # combine shared and class specific outputs
            conv_outs = shared_conv_outs + all_class_conv_outs[i]
            # first deconv layer has no skip connection
            x2 = layer_list[0](conv_outs[-1])
            # forward pass on remaining deconv layers
            for j, layer in enumerate(layer_list[1:]):
                reverse_layer_number = self.n_conv_layers - (j + 2)
                sc = conv_outs[reverse_layer_number]
                x2 = layer(torch.cat([x2, sc], dim=1))
            x3 += [x2]
        # batch size x nclasses x height x width
        x3 = torch.stack(x3, dim=1)

        return x3

    def forward(self, x):
        """Forward Pass."""
        x1, shared_conv_outs = self._shared_convolutions(x)
        all_class_conv_outs = self._class_convolutions(x1)
        out = self._class_deconvolutions(shared_conv_outs, all_class_conv_outs)
        mask = self.sigmoid(out)

        return x.unsqueeze(1).expand_as(mask) * mask, mask
