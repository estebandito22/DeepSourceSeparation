"""Script for evaluating trained source separation model on musdb18."""

from collections import defaultdict
from argparse import ArgumentParser
import functools

import musdb
import museval

import numpy as np
import pandas as pd
from tqdm import tqdm
from librosa.core import istft
from librosa.core import resample

import torch
from torch.utils.data import DataLoader

from dss.datasets.bandhubeval import BandhubEvalset
from dss.trainers.mhmmdensenetlstm import MHMMDenseNetLSTM


if __name__ == '__main__':
    """
    Usage:
        python test_densenet.py \
            --metadata_path metadata/musdb18_STFT.csv \
            --model_dir dense_models/model_dir \
            --epoch 500 \
            --split test
    """

    ap = ArgumentParser()
    ap.add_argument("-mp", "--metadata_path",
                    help="Location of metadata for evaluation.")
    ap.add_argument("-md", "--model_dir",
                    help="Location of pretrained model.")
    ap.add_argument("-ep", "--epoch",
                    help="epoch of pretrained model.")
    ap.add_argument("-sp", "--split",
                    help="Split to evaluate on.")
    ap.add_argument("-sd", "--save_dir",
                    help="Save framewise evaluations to directory.")
    args = vars(ap.parse_args())

    print("\n\nLoading " + args['metadata_path'] + "\n")

    print("Evaluating " + args['model_dir']
          + " on epoch " + args['epoch'] + " for split " + args['split']
          + "\n\n")

    print("Saving to " + str(args['save_dir']) + "\n\n")

    df = pd.read_csv(args['metadata_path'])

    dnet = MHMMDenseNetLSTM()
    dnet.load(args['model_dir'], args['epoch'])

    evalset = BandhubEvalset(
        df, args['split'], n_frames=dnet.n_frames, harmonics=dnet.harmonics,
        random_seed=0)

    eval_loader = DataLoader(
        evalset, batch_size=1, shuffle=False,
        num_workers=8, collate_fn=evalset.collate_func)

    sdr, sir, sar, _ = dnet.score(eval_loader, framewise=True,
                               save_dir=args['save_dir'])
    print("SDR", sdr)
    print("SIR", sir)
    print("SAR", sar)
