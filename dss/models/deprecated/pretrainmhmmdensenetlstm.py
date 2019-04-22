"""Class constructing UNet model for source separation task."""

import torch
from dss.models.mhmmdensenetlstm import MHMMDenseNetLSTMModel


class PretrainMHMMDenseNetLSTMModel(MHMMDenseNetLSTMModel):

    """Multi-Head MMDenseNetLSTMModel source separation model."""

    def __init__(self, n_classes=1, n_shared_layers=3, in_channels=1,
                 hidden_size=512, batch_size=64):
        """Initialize PretrainMHMMDenseNetLSTMModel."""
        MHMMDenseNetLSTMModel.__init__(
            self, n_classes=n_classes, n_shared_layers=n_shared_layers,
            in_channels=in_channels, hidden_size=hidden_size,
            batch_size=batch_size)

    def forward(self, x):
        """Forward Pass."""
        out_low = self._band_forward(
            x[:, :, :384, :], self.shared_layers_low, self.class_layers_low)
        out_high = self._band_forward(
            x[:, :, 384:, :], self.shared_layers_high, self.class_layers_high)
        out_full = self._band_forward(
            x, self.shared_layers_full, self.class_layers_full)

        out_highlow = [torch.cat([low, high], dim=2)
                       for low, high in zip(out_low, out_high)]
        out_all = [torch.cat([full, highlow], dim=1)
                   for full, highlow in zip(out_full, out_highlow)]

        out = self._class_final_dense(out_all)

        return out.squeeze(1)
