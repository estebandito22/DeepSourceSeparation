"""Script for training Source Separation Unet."""

from argparse import ArgumentParser
import pandas as pd
import torch

from dss.datasets.bandhubsimple import BandhubDataset
from dss.datasets.bandhubpred import BandhubPredset
# from dss.datasets.bandhubeval import BandhubEvalset
from dss.trainers.mhmmdensenetlstm import MHMMDenseNetLSTM


if __name__ == '__main__':
    """
    Usage:
        python train.py \
            --n_shared_layers 3 \
            --in_channels 1 \
            --hidden_size 512 \
            --batch_size 4 \
            --learning_rate 0.001 \
            --weight_decay 0.0 \
            --num_epochs 100 \
            --metadata_path metadata/tracks10kMSDfake.csv \
            --save_dir unet_models
    """

    ap = ArgumentParser()
    ap.add_argument("-sl", "--n_shared_layers", type=int, default=3,
                    help="Number of shared layers in UNet.")
    ap.add_argument("-ic", "--in_channels", type=int, default=1,
                    help="Input channels to UNet.")
    ap.add_argument("-hs", "--hidden_size", type=int, default=512,
                    help="Hidden size for LSTM.")
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
    ap.add_argument("-pp", "--pretrained_path",
                    help="Location of pretrained model.")
    args = vars(ap.parse_args())

    df = pd.read_csv(args['metadata_path'])
    # df = df[df['instrument'].isin([2, 7, 0, 3, 5])]
    # df['instrument'] = df['instrument'].astype('category').cat.codes
    classes = df['instrument'].unique()
    train_datasets = BandhubDataset(df, 'train')
    val_datasets = BandhubDataset(df, 'val', random_seed=0)
    train_predset = BandhubPredset(df, 'train', random_seed=0)
    val_predset = BandhubPredset(df, 'val', random_seed=0)

    if args['pretrained_path'] is not None:
        with open(args['pretrained_path'], 'rb') as model_dict:
            if torch.cuda.is_available():
                checkpoint = torch.load(model_dict)
            else:
                checkpoint = torch.load(model_dict, map_location='cpu')
        pretrained_state = checkpoint['state_dict']
    else:
        pretrained_state = None

    dnet = MHMMDenseNetLSTM(n_classes=len(classes),
                            n_shared_layers=args['n_shared_layers'],
                            in_channels=args['in_channels'],
                            hidden_size=args['hidden_size'],
                            batch_size=args['batch_size'],
                            lr=args['learning_rate'],
                            weight_decay=args['weight_decay'],
                            num_epochs=args['num_epochs'],
                            objective=args['objective'])

    dnet.fit(train_datasets, val_datasets, train_predset, val_predset,
             args['save_dir'], pretrained_state)
