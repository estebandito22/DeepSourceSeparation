"""Class constructing UNet model for source separation task."""

from dss.models.mhunet import MHUNetModel


class PretrainMHUNetModel(MHUNetModel):

    """Mutli-Head UNet source separation model."""

    def __init__(self, n_classes=1, n_shared_layers=3, in_channels=1,
                 negative_slope=0.2, dropout=0.5):
        """Initialize MutliHeadUnetModel."""
        MHUNetModel.__init__(
            self, n_classes=n_classes, n_shared_layers=n_shared_layers,
            in_channels=in_channels, negative_slope=negative_slope,
            dropout=dropout)

    def forward(self, x):
        """Forward Pass."""
        x1, shared_conv_outs = self._shared_convolutions(x)
        all_class_conv_outs = self._class_convolutions(x1)
        out = self._class_deconvolutions(shared_conv_outs, all_class_conv_outs)
        return out.squeeze(1)
