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
    ap.add_argument("-of", "--offset", type=int, default=0,
                    help="Number of shared layers offset for full band.")
    ap.add_argument("-ic", "--in_channels", type=int, default=1,
                    help="Input channels to UNet.")
    ap.add_argument("-ks", "--kernel_size", type=int, default=3,
                    help="Kernel size for convolutional layers.")
    ap.add_argument("-hs", "--hidden_size", type=int, default=512,
                    help="Hidden size for LSTM.")
    ap.add_argument("-la", "--loss_alphas", action='store_true',
                    help="Use loss alphas.")
    ap.add_argument("-nm", "--normalize_masks", action='store_true',
                    help="Normalize predicted masks.")
    ap.add_argument("-us", "--upper_bound_slope", type=float, default=1/100,
                    help="Slope of the upper bound curriculum.")
    ap.add_argument("-uv", "--uniform_volumes", action='store_true',
                    help="Use random uniform volume data augmentation.")
    ap.add_argument("-im", "--instrument_mask", type=float, default=0.0,
                    help="Instrument mask rate.")
    ap.add_argument("-msc", "--mask_curriculum", type=float, default=0.0,
                    help="Epoch to reach instrument mask level.")
    ap.add_argument("-gs", "--gain_slope", type=float, default=0.0,
                    help="Gain modulation slope.")
    ap.add_argument("-if", "--interference", type=float, default=0.1,
                    help="Add random signal interference.")
    ap.add_argument("-ls", "--lower_bound_slope", type=float, default=1/400,
                    help="Slope of the lower bound curriculum.")
    ap.add_argument("-mf", "--mag_func", default='sqrt',
                    help="Function to use on spectrograms, 'sqrt' or 'log'.")
    ap.add_argument("-nf", "--n_frames", type=int, default=512,
                    help="Number of samples to use in the time domain.")
    ap.add_argument("-th", "--threshold", type=float,
                    help="Threshold below which volume alphas are set to 0.0.")
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
    ap.add_argument("-rg", "--regression", action='store_true',
                    help="Perform regression instead of masking.")
    ap.add_argument("-mp", "--metadata_path",
                    help="Location of metadata for training.")
    ap.add_argument("-sd", "--save_dir",
                    help="Location to save the model.")
    ap.add_argument("-pp", "--pretrained_path",
                    help="Location of pretrained model.")
    ap.add_argument("-ev", "--eval_version",
                    help="Vertsion of BSS to use for evaluation.")
    ap.add_argument("-tc", "--train_class", type=int,
                    help="Specific class to train. -1 is all -999 is random.")
    ap.add_argument("-nfft", "--n_fft", type=int, default=1025,
                    help="Number of FFT corresponds to input frequency.")
    ap.add_argument("-cs", "--chan_swap", type=float, default=0.0,
                    help="Mean of Bernoulli for channel swapping.")
    ap.add_argument("-cp", "--continue_path",
                    help="Path to model for warm start.")
    ap.add_argument("-ce", "--continue_epoch", type=int,
                    help="Epoch of model for ward start.")
    args = vars(ap.parse_args())

    print("\n\nLoading " + args['metadata_path'] + "\n\n")

    df = pd.read_csv(args['metadata_path'])

    if args['save_dir'].find('medley') > -1:
        df = df[df['instrument'].isin([0, 1, 2, 3, 4, 6, 5, 7, 8, 9])]
        df['instrument'] = df['instrument'].astype('category').cat.codes

    classes = df['instrument'].unique()

    train_datasets = BandhubDataset(
        df, 'train', mag_func=args['mag_func'], n_frames=args['n_frames'],
        upper_bound_slope=args['upper_bound_slope'],
        lower_bound_slope=args['lower_bound_slope'],
        threshold=args['threshold'],
        chan_swap=args['chan_swap'],
        uniform_volumes=args['uniform_volumes'],
        interference=args['interference'],
        instrument_mask=args['instrument_mask'],
        gain_slope=args['gain_slope'],
        mask_curriculum=args['mask_curriculum'])

    val_datasets = BandhubDataset(
        df, 'val', mag_func=args['mag_func'], n_frames=args['n_frames'],
        concentration=1000., random_seed=0)

    train_predset = BandhubPredset(
        df, 'train', mag_func=args['mag_func'], n_frames=args['n_frames'],
        random_seed=0)

    val_predset = BandhubPredset(
        df, 'val', mag_func=args['mag_func'], n_frames=args['n_frames'],
        random_seed=0)

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
                            kernel_size=args['kernel_size'],
                            hidden_size=args['hidden_size'],
                            loss_alphas=args['loss_alphas'],
                            normalize_masks=args['normalize_masks'],
                            batch_size=args['batch_size'],
                            lr=args['learning_rate'],
                            weight_decay=args['weight_decay'],
                            num_epochs=args['num_epochs'],
                            objective=args['objective'],
                            eval_version=args['eval_version'],
                            train_class=args['train_class'],
                            n_fft=args['n_fft'],
                            regression=args['regression'],
                            offset=args['offset'])

    if args['continue_path'] and args['continue_epoch']:
        dnet.load(args['continue_path'], args['continue_epoch'])
        warm_start = True
    else:
        warm_start = False

    dnet.fit(train_datasets, val_datasets, train_predset, val_predset,
             args['save_dir'], pretrained_state, warm_start)
