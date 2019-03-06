"""Script for pretraining Unet."""

from argparse import ArgumentParser
import pandas as pd

from dss.datasets.msd import MSDDataset
from dss.trainers.pretrainmhunet import PretrainMultiHeadUNet


if __name__ == '__main__':
    """
    Usage:
        python pretrain.py \
            --negative_slope 0.2 \
            --dropout 0.5 \
            --batch_size 4 \
            --learning_rate 0.001 \
            --weight_decay 0.0 \
            --num_epochs 100 \
            --metadata_path metadata/tracks10kMSDfake.csv \
            --save_dir pretrain_unet_models
    """

    ap = ArgumentParser()
    ap.add_argument("-ic", "--in_channels", type=int, default=1,
                    help="Input channels to UNet.")
    ap.add_argument("-ns", "--negative_slope", type=float, default=0.2,
                    help="Negative slope for LeakyReLU.")
    ap.add_argument("-do", "--dropout", type=float, default=0.5,
                    help="Dropout in UNet.")
    ap.add_argument("-bs", "--batch_size", type=int, default=2,
                    help="Batch size for optimization.")
    ap.add_argument("-lr", "--learning_rate", type=float, default=0.001,
                    help="Learning rate for optimization.")
    ap.add_argument("-wd", "--weight_decay", type=float, default=0.0,
                    help="Weight decay for optimization.")
    ap.add_argument("-ne", "--num_epochs", type=int, default=100,
                    help="Number of epochs for optimization.")
    ap.add_argument("-ob", "--objective", default='L1',
                    help="Objective function to use, 'L1' or 'L2'.")
    ap.add_argument("-mp", "--metadata_path",
                    help="Location of metadata for training.")
    ap.add_argument("-sd", "--save_dir",
                    help="Location to save the model.")
    args = vars(ap.parse_args())

    # TODO: Remove degenerate inputs

    df = pd.read_csv(args['metadata_path'])
    train_dataset = MSDDataset(df, 'train')
    val_dataset = MSDDataset(df, 'val')

    unet = PretrainMultiHeadUNet(n_classes=1,
                                 n_shared_layers=3,
                                 in_channels=args['in_channels'],
                                 negative_slope=args['negative_slope'],
                                 dropout=args['dropout'],
                                 batch_size=args['batch_size'],
                                 lr=args['learning_rate'],
                                 weight_decay=args['weight_decay'],
                                 num_epochs=args['num_epochs'],
                                 objective=args['objective'])

    unet.fit(train_dataset, val_dataset, args['save_dir'])
