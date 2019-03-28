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
    args = vars(ap.parse_args())

    print("\n\nLoading " + args['metadata_path'] + "\n")

    print("Evaluating " + args['model_dir']
          + " on epoch " + args['epoch'] + "\n")

    df = pd.read_csv(args['metadata_path'])

    evalset = BandhubEvalset(df, args['split'], random_seed=0)

    eval_loader = DataLoader(
        evalset, batch_size=2, shuffle=False,
        num_workers=8, collate_fn=evalset.collate_func)

    dnet = MHMMDenseNetLSTM()
    dnet.load(args['model_dir'], args['epoch'])

    sdr, sir, sar = dnet.score(eval_loader, framewise=True)
    print("SDR", sdr)
    print("SIR", sir)
    print("SAR", sar)
