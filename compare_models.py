"""Script to compare models."""

import os
import json
import glob
from collections import defaultdict
from argparse import ArgumentParser

import numpy as np
from scipy.stats import ranksums, wilcoxon
from scipy.stats import ttest_ind

from dss.utils.utils import medians, means, song_means, get_metric_array, \
    combine_metric_arrays


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument("-pm", "--previous_model", default='TAK1',
                    help="The name of the previously published model.")
    ap.add_argument("-om", "--our_model", default='framewise_out_musdb_final/test',
                    help="The location of our final results.")
    ap.add_argument("-sd", "--save_dir",
                    help="Location to save outputs.")
    args = vars(ap.parse_args())

    # load other model data
    prev_model_folder = os.path.join(
        '../', 'sigsep-mus-2018-master', 'submissions', args['previous_model'],
        'test')
    prev_model_files = glob.glob(prev_model_folder + '/*.json')
    prev_model_datas = []
    for file in prev_model_files:
        with open(file, 'r') as f:
            prev_model_data = json.load(f)
            prev_model_datas += [prev_model_data]

    # load our model data
    our_model_files = glob.glob(args['our_model'] + '/*.json')
    our_model_datas = []
    for file in our_model_files:
        with open(file, 'r') as f:
            our_model_data = json.load(f)
            our_model_datas += [our_model_data]

    # compute medians and save
    prev_model_medians = medians(prev_model_datas)
    our_model_medians = medians(our_model_datas)

    save_file = os.path.join(
        args['save_dir'], args['previous_model'] + '_medians.json')
    with open(save_file, 'w') as f:
        json.dump(prev_model_medians, f)

    save_file = os.path.join(
        args['save_dir'], 'MHMMDenseLSTM_medians.json')
    with open(save_file, 'w') as f:
        json.dump(our_model_medians, f)

    # compute medians and save
    prev_model_means = means(prev_model_datas)
    our_model_means = means(our_model_datas)

    save_file = os.path.join(
        args['save_dir'], args['previous_model'] + '_means.json')
    with open(save_file, 'w') as f:
        json.dump(prev_model_means, f)

    save_file = os.path.join(
        args['save_dir'], 'MHMMDenseLSTM_means.json')
    with open(save_file, 'w') as f:
        json.dump(our_model_means, f)

    # generate combined arrasy
    prev_model_combined_sdr = combine_metric_arrays(prev_model_datas, 'SDR')
    our_model_combined_sdr = combine_metric_arrays(our_model_datas, 'SDR')

    prev_model_song_sdr = song_means(prev_model_datas)
    our_model_song_sdr = song_means(our_model_datas)

    # compute rank sum, nan and save
    rank_sum_res_frame = defaultdict(list)
    rank_sum_res_song = defaultdict(list)
    signed_rank_res_frame = defaultdict(list)
    signed_rank_res_song = defaultdict(list)
    # ttest_res_frame = defaultdict(list)
    # ttest_res_song = defaultdict(list)
    prev_model_nan_res = defaultdict(list)
    our_model_nan_res = defaultdict(list)
    for k in our_model_combined_sdr.keys():
        if k == 'accompaniment':
            continue
        rank_sum_res_frame[k] = wilcoxon(
            our_model_combined_sdr[k], prev_model_combined_sdr[k])
        rank_sum_res_song[k] = wilcoxon(
            our_model_song_sdr[k], prev_model_song_sdr[k])
        rank_sum_res_frame[k] = ranksums(
            our_model_combined_sdr[k], prev_model_combined_sdr[k])
        rank_sum_res_song[k] = ranksums(
            our_model_song_sdr[k], prev_model_song_sdr[k])
        # ttest_res_frame[k] = ttest_ind(
        #     our_model_combined_sdr[k], prev_model_combined_sdr[k])
        # ttest_res_song[k] = ttest_ind(
        #     our_model_song_sdr[k], prev_model_song_sdr[k])
        prev_model_nan_res[k] = np.isnan(prev_model_combined_sdr[k]).sum() \
            / len(prev_model_combined_sdr[k])
        our_model_nan_res[k] = np.isnan(our_model_combined_sdr[k]).sum() \
            / len(our_model_combined_sdr[k])

    save_file = os.path.join(
        args['save_dir'],
        'MHMMDenseLSTM_' + args['previous_model'] + '_wilcoxon_framewise.json')
    with open(save_file, 'w') as f:
        json.dump(signed_rank_res_frame, f)

    save_file = os.path.join(
        args['save_dir'],
        'MHMMDenseLSTM_' + args['previous_model'] + '_wilcoxon_song.json')
    with open(save_file, 'w') as f:
        json.dump(signed_rank_res_song, f)

    save_file = os.path.join(
        args['save_dir'],
        'MHMMDenseLSTM_' + args['previous_model'] + '_ranksum_framewise.json')
    with open(save_file, 'w') as f:
        json.dump(rank_sum_res_frame, f)

    save_file = os.path.join(
        args['save_dir'],
        'MHMMDenseLSTM_' + args['previous_model'] + '_ranksum_song.json')
    with open(save_file, 'w') as f:
        json.dump(rank_sum_res_song, f)

    # save_file = os.path.join(
    #     args['save_dir'],
    #     'MHMMDenseLSTM_' + args['previous_model'] + '_ttest_framewise.json')
    # with open(save_file, 'w') as f:
    #     json.dump(ttest_res_frame, f)
    #
    # save_file = os.path.join(
    #     args['save_dir'],
    #     'MHMMDenseLSTM_' + args['previous_model'] + '_ttest_song.json')
    # with open(save_file, 'w') as f:
    #     json.dump(ttest_res_song, f)

    save_file = os.path.join(
        args['save_dir'], args['previous_model'] + '_nan.json')
    with open(save_file, 'w') as f:
        json.dump(prev_model_nan_res, f)

    save_file = os.path.join(
        args['save_dir'], 'MHMMDenseLSTM_nan.json')
    with open(save_file, 'w') as f:
        json.dump(our_model_nan_res, f)
