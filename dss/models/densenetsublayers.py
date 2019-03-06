"""Sublayers for DenseNet."""

import torch
import torch.nn as nn


class DenseBlock(nn.Module):

    """Dense block."""

    def __init__(self, in_channels, out_channels=12, kernel_size=3, nlayers=1):
        """Initialize Dense block."""
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.nlayers = nlayers
        super(DenseBlock, self).__init__()

        assert self.kernel_size % 2 == 1, "kernel_size must be odd."

        self.layers = nn.ModuleList(
            [self._build_layer(i) for i in range(self.nlayers)])

    def _build_layer(self, layer):
        layer = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels + self.out_channels * layer,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size, padding=self.kernel_size//2),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU())
        return layer

    def forward(self, x):
        """Forward Pass."""
        x_outs = [x]
        for layer in self.layers:
            x_outs += [layer(torch.cat(x_outs, dim=1))]
        return x_outs[-1]


class DownSampleBlock(nn.Module):

    """DownSampleBlock."""

    def __init__(self, channels, kernel_size=2):
        """Initialize DownSampleBlock."""
        self.channels = channels
        self.kernel_size = kernel_size
        super(DownSampleBlock, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=self.channels, out_channels=self.channels,
            kernel_size=1)
        self.pool = nn.AvgPool2d(kernel_size=self.kernel_size)

    def forward(self, x):
        """Forward Pass."""
        x = self.conv(x)
        return self.pool(x)


class LSTMBlock(nn.Module):

    """LSTMBlock."""

    def __init__(self, in_channels, in_size, batch_size, hidden_size=512):
        """Initialize LSTMBlock."""
        self.in_channels = in_channels
        self.in_size = in_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.hidden = None
        super(LSTMBlock, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=self.in_channels, out_channels=1, kernel_size=1)
        self.lstm = nn.LSTM(
            input_size=self.in_size, hidden_size=self.hidden_size,
            bidirectional=True)
        self.init_hidden(self.batch_size)
        self.linear = nn.Linear(self.hidden_size * 2, self.in_size)

    def init_hidden(self, batch_size):
        """Initialize the hidden state of the RNN."""
        hidden1 = torch.zeros(2, batch_size, self.hidden_size)
        hidden2 = torch.zeros(2, batch_size, self.hidden_size)
        if torch.cuda.is_available():
            hidden1 = hidden1.cuda()
            hidden2 = hidden2.cuda()
        self.hidden = (hidden1, hidden2)

    def detach_hidden(self, batch_size):
        """Detach the hidden state of the RNN."""
        hidden, c_t = self.hidden
        _, hidden_batch_size, _ = hidden.size()

        if hidden_batch_size != batch_size:
            self.init_hidden(batch_size)
        else:
            detached_c_t = c_t.detach()
            detached_c_t.zero_()
            detached_hidden = hidden.detach()
            detached_hidden.zero_()
            self.hidden = (detached_hidden, detached_c_t)

    def forward(self, x):
        """Forward Pass."""
        x = self.conv(x).squeeze(1)
        # batch_size x freq x time
        x, _ = self.lstm(x.permute(2, 0, 1), self.hidden)
        # time x batch_size x hidden_size
        x = self.linear(x.permute(1, 0, 2))
        # batch_size x time x freq
        return x.permute(0, 2, 1).unsqueeze(1)


class DenseLSTMBlock(nn.Module):

    """DenseBlock followed by LSTMBlock and concat."""

    def __init__(self, in_channels, out_channels, kernel_size, nlayers,
                 in_size, batch_size, hidden_size):
        """Initialize DenseLSTMBlock."""
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.nlayers = nlayers
        self.in_size = in_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        super(DenseLSTMBlock, self).__init__()

        self.dense = DenseBlock(
            in_channels=self.in_channels, out_channels=self.out_channels,
            kernel_size=self.kernel_size, nlayers=self.nlayers)

        self.lstm = LSTMBlock(
            in_channels=self.out_channels, in_size=self.in_size,
            batch_size=self.batch_size, hidden_size=self.hidden_size)

    def forward(self, x):
        """Forward pass."""
        x = self.dense(x)
        x1 = self.lstm(x)
        return torch.cat([x, x1], dim=1)
