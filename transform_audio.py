"""Script to transform audio to complex spectrogram."""

from argparse import ArgumentParser
from multiprocessing import Process
import multiprocessing as mp
import pandas as pd

from preprocessing.stfttensor import STFTTransformer


def main(meta_file, save_dir, overwrite, n_fft, hop_length, groups,
         group, out_q, mono, sample_rate):
    """Execute stft transform."""
    stft = STFTTransformer(n_fft, hop_length, mono, sample_rate)
    stft.transform(
        meta_file, save_dir, overwrite, groups, group, out_q)


if __name__ == '__main__':

    ap = ArgumentParser()
    ap.add_argument("-s", "--save_dir",
                    help="The directory where spectrograms are saved.")
    ap.add_argument("-mf", "--meta_file",
                    help="The full path to the metadata file to use.")
    ap.add_argument("-m", "--meta_loc",
                    help="The full path to the metadata file to save.")
    ap.add_argument("-o", "--overwrite", action='store_true',
                    help="Should previously transformed data be overwritten.")
    ap.add_argument("-n", "--n_fft", default=2048, type=int,
                    help="n_fft to use in melspectrogram transform.")
    ap.add_argument("-hl", "--hop_length", default=512, type=int,
                    help="Hop length to use in melspectrogram transform.")
    ap.add_argument("-mo", "--mono", action='store_true',
                    help="Convert to mono.")
    ap.add_argument("-sr", "--sample_rate", type=int, default=22050,
                    help="Sample rate for loading audio.")
    args = vars(ap.parse_args())

    sd = args['save_dir']
    mf = args['meta_file']
    ml = args['meta_loc']
    ov = args["overwrite"]
    nf = args["n_fft"]
    hl = args["hop_length"]
    mo = args["mono"]
    sr = args["sample_rate"]

    r = mp.cpu_count()
    o_q = mp.Queue()

    processes = []
    for i in range(r):
        p = Process(
            target=main, args=(mf, sd, ov, nf, hl, r, i, o_q, mo, sr))
        p.start()
        processes.append(p)

    resultdict = {'stft_path': [], 'trackId': [], 'songId': [],
                  'instrument': [], 'trackVolume': [], 'urlId': []}
    for i in range(r):
        q_dict = o_q.get()
        resultdict['stft_path'] += q_dict['stft_path']
        resultdict['trackId'] += q_dict['trackId']
        resultdict['songId'] += q_dict['songId']
        resultdict['instrument'] += q_dict['instrument']
        resultdict['trackVolume'] += q_dict['trackVolume']
        resultdict['urlId'] += q_dict['urlId']

    for p in processes:
        p.join()

    pd.DataFrame(resultdict).to_csv(ml, index=False)
