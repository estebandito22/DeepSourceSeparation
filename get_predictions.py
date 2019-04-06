"""Script for training Source Separation Unet."""

import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from tqdm import tqdm
from librosa.core import istft
from scipy.signal import wiener

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from dss.datasets.bandhubeval import BandhubEvalset
from dss.trainers.mhmmdensenetlstm import MHMMDenseNetLSTM


USE_CUDA = torch.cuda.is_available()


def get_preds_recons(model, loader):
    """
    Get predicted magnitudes and reconstructios along with targets.

    Args
    ----
        model : SourceSeparator model.
        loader : PyTorch DataLoader.

    """
    model.model.eval()

    pred_recons_out = []
    y_recons_out = []
    pred_mags_out = []
    y_mags_out = []
    cs_out = []
    pred_mask_out = []
    pred_cmplx_out = []
    y_cmplx_out = []

    # list of batches
    preds, ys, cs, ts, ms = model.predict(loader)

    # # only perform framewise evaluation at testing time
    if model.n_fft == 1025:
        rate = 22050
        hop = 512
        win = 2048
    elif model.n_fft == 2049:
        rate = 44100
        hop = 1024
        win = 4096
    if not framewise:
        rate = np.inf

    # for each batch
    for b_preds, b_ys, b_cs, b_ts, b_ms in tqdm(
            list(zip(preds, ys, cs, ts, ms))):
        # for each sample
        for pred, y, c, t, m in zip(b_preds, b_ys, b_cs, b_ts, b_ms):
            pred_recons = []
            y_recons = []
            pred_mags = []
            y_mags = []
            pred_cs = []
            pred_mask = []
            pred_cmplx = []
            y_cmplx = []
            # for each class
            for i, (c_pred, c_y, c_c, c_m) in enumerate(zip(pred, y, c, m)):
                # if the class exists in the source signal
                if c_c == 1 and np.abs(c_y).sum() > 0:
                    c_pred = c_pred[..., :t]
                    c_y = c_y[..., :t]
                    c_m = c_m[..., :t]
                    pred_recon = []
                    y_recon = []
                    for c_pred_chan, c_y_chan in zip(c_pred, c_y):
                        pred_recon += [istft(
                            c_pred_chan, hop_length=hop, win_length=win)]
                        y_recon += [istft(
                            c_y_chan, hop_length=hop, win_length=win)]
                    pred_recon = np.stack(pred_recon, axis=-1)
                    y_recon = np.stack(y_recon, axis=-1)
                    pred_recons += [pred_recon]
                    y_recons += [y_recon]
                    pred_mags += [np.abs(c_pred)]
                    y_mags += [np.abs(c_y)]
                    pred_cs += [i]
                    pred_mask += [c_m]
                    pred_cmplx += [c_pred]
                    y_cmplx += [c_y]
            # possible to sample from targets that are all zeros
            if pred_recons:
                pred_recons = np.stack(pred_recons)
                y_recons = np.stack(y_recons)
                pred_mags = np.stack(pred_mags)
                y_mags = np.stack(y_mags)
                pred_cs = np.stack(pred_cs)
                pred_mask = np.stack(pred_mask)
                pred_cmplx = np.stack(pred_cmplx)
                y_cmplx = np.stack(y_cmplx)

                pred_recons_out += [pred_recons]
                y_recons_out += [y_recons]
                pred_mags_out += [pred_mags]
                y_mags_out += [y_mags]
                cs_out += [pred_cs]
                pred_mask_out += [pred_mask]
                pred_cmplx_out += [pred_cmplx]
                y_cmplx_out += [y_cmplx]

    return pred_recons_out, y_recons_out, pred_mags_out, y_mags_out, cs_out, \
        pred_mask_out, pred_cmplx_out, y_cmplx


if __name__ == '__main__':
    """
    Usage:
        python analyze_predictions.py \
            --metadata_path metadata/dsd100.csv \
            --save_dir analysis \
            --model_dir dense_models \
            --epoch 500
    """

    ap = ArgumentParser()
    ap.add_argument("-mp", "--metadata_path",
                    help="Location of metadata for training.")
    ap.add_argument("-sd", "--save_dir",
                    help="Location to save the model.")
    ap.add_argument("-md", "--model_dir",
                    help="Location of pretrained model.")
    ap.add_argument("-ep", "--epoch",
                    help="epoch of pretrained model.")
    ap.add_argument("-sm", "--sample", type=int,
                    help="Which sample to analyze.")
    args = vars(ap.parse_args())

    df = pd.read_csv(args['metadata_path'])

    val_predset = BandhubEvalset(df, 'val', random_seed=0)
    val_subset = Subset(val_predset, [args['sample']])

    val_pred_loader = DataLoader(
        val_subset, batch_size=1, shuffle=False,
        num_workers=8, collate_fn=val_predset.collate_func)

    dnet = MHMMDenseNetLSTM()
    dnet.load(args['model_dir'], args['epoch'])

    pred_recon, y_recon, pred_mag, y_mag, pred_c, pred_mask, pred_cmplx, y_cmplx = \
        get_preds_recons(dnet, val_pred_loader)

    f = os.path.join(
        args['save_dir'], 'sourcesep_pred_recon{}.npy'.format(args['sample']))
    np.save(f, pred_recon)
    f = os.path.join(
        args['save_dir'], 'sourcesep_y_recon{}.npy'.format(args['sample']))
    np.save(f, y_recon)
    f = os.path.join(
        args['save_dir'], 'sourcesep_pred_mag{}.npy'.format(args['sample']))
    np.save(f, pred_mag)
    f = os.path.join(
        args['save_dir'], 'sourcesep_y_mag{}.npy'.format(args['sample']))
    np.save(f, y_mag)
    f = os.path.join(
        args['save_dir'], 'sourcesep_pred_c{}.npy'.format(args['sample']))
    np.save(f, pred_c)
    f = os.path.join(
        args['save_dir'], 'sourcesep_pred_mask{}.npy'.format(args['sample']))
    np.save(f, pred_mask)
    f = os.path.join(
        args['save_dir'], 'sourcesep_pred_cmplx{}.npy'.format(args['sample']))
    np.save(f, pred_cmplx)
    f = os.path.join(
        args['save_dir'], 'sourcesep_y_cmplx{}.npy'.format(args['sample']))
    np.save(f, y_cmplx)
