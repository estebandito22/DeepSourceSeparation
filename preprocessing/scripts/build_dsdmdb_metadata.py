import numpy as np
import pandas as pd
import librosa
import os
import glob
import uuid
from collections import defaultdict

import medleydb as mdb


CWD = os.getcwd()
DATA = os.path.join(CWD, 'data', 'DSD100', 'Sources', 'Dev')

data = defaultdict(list)
file_counts = []
for root, dirs, files in os.walk(DATA, topdown=False):
    for file in files:
        data['file_path'] += [os.path.join(root, file)]
        data['trackId'] += [str(uuid.uuid4())]
        data['instrument'] += [file.split('.')[0]]
        data['trackVolume'] += [1]
        file_counts += [len(files)]
    if dirs:
        for d, fcount in zip(dirs, file_counts):
            songId = str(uuid.uuid4())
            for _ in range(fcount):
                data['songId'] += [songId]


DATA = os.path.join(CWD, 'data', 'DSD100', 'Sources', 'Test')

file_counts = []
for root, dirs, files in os.walk(DATA, topdown=False):
    for file in files:
        data['file_path'] += [os.path.join(root, file)]
        data['trackId'] += [str(uuid.uuid4())]
        data['instrument'] += [file.split('.')[0]]
        data['trackVolume'] += [1]
        file_counts += [len(files)]
    if dirs:
        for d, fcount in zip(dirs, file_counts):
            songId = str(uuid.uuid4())
            for _ in range(fcount):
                data['songId'] += [songId]


mtrack_generator = mdb.load_all_multitracks()
common_path = '/scratch/swc419/DeepSourceSeparation/data/V1/'
for mtrack in mtrack_generator:
    track_id = mtrack.track_id
    track_uuid = str(uuid.uuid4())
    zipped_data = zip(mtrack.stem_instruments,
                      list(mtrack.mixing_coefficients['stft'].values()))
    for i, (instrument, volume) in enumerate(zipped_data):
        file_path = os.path.join(
            common_path, track_id,
            track_id + '_STEMS', track_id + '_STEM_{:02d}'.format(i+1) + '.wav')
        data['file_path'].append(file_path)
        data['songId'].append(track_uuid)
        data['trackId'].append(str(uuid.uuid4()))
        data['instrument'].append(instrument)
        data['trackVolume'].append(volume)

instrument_map = {'clean electric guitar': 'guitar',
                 'distorted electric guitar': 'guitar',
                 'drum set': 'drums',
                 'electric bass': 'bass',
                 'fx/processed sound': 'other',
                 'male singer': 'vocals',
                 'synthesizer': 'other',
                 'tambourine': 'other',
                 'vocalists': 'vocals',
                 'acoustic guitar': 'guitar',
                 'female singer': 'vocals',
                 'cello': 'strings',
                 'clarinet': 'wind',
                 'trombone': 'brass',
                 'brass section': 'brass',
                 'harmonica': 'other',
                 'viola section': 'strings',
                 'string section': 'strings',
                 'auxiliary percussion': 'other',
                 'banjo': 'other',
                 'horn section': 'brass',
                 'vibraphone': 'other',
                 'male speaker': 'vocals',
                 'tack piano': 'piano',
                 'drum machine': 'drums',
                 'mandolin': 'other',
                 'tabla': 'other',
                 'kick drum': 'drums',
                 'lap steel guitar': 'other',
                 'timpani': 'other',
                 'snare drum': 'drums',
                 'tenor saxophone': 'brass',
                 'glockenspiel': 'other',
                 'double bass': 'drums',
                 'melodica': 'other',
                 'male rapper': 'vocals',
                 'sampler': 'other',
                 'cymbal': 'other',
                 'violin section': 'strings',
                 'claps': 'other',
                 'shaker': 'other',
                 'trumpet section': 'brass',
                 'bassoon': 'wind',
                 'chimes': 'other',
                 'flute': 'wind',
                 'french horn': 'brass',
                 'trumpet': 'brass',
                 'viola': 'strings',
                 'violin': 'strings',
                 'Main System': 'other',
                 'darbuka': 'other',
                 'doumbek': 'other',
                 'oud': 'other',
                 'scratches': 'other',
                 'bongo': 'other',
                 'french horn section': 'brass',
                 'harp': 'other',
                 'oboe': 'wind',
                 'bass drum': 'drums',
                 'cello section': 'strings',
                 'clarinet section': 'wind',
                 'gong': 'other',
                 'bamboo flute': 'wind',
                 'flute section': 'wind',
                 'piccolo': 'wind',
                 'trombone section': 'brass',
                 'dizi': 'other',
                 'erhu': 'other',
                 'guzheng': 'other',
                 'yangqin': 'other',
                 'zhongruan': 'other',
                 'liuqin': 'other',
                 'gu': 'other',
                 'baritone saxophone': 'wind',
                 'bass clarinet': 'wind',
                 'alto saxophone': 'wind',
                 'electric piano': 'piano',
                 'soprano saxophone': 'wind',
                 'tuba': 'brass',
                 'toms': 'other',
                 'accordion': 'other'}

df = pd.DataFrame(data)
df['instrument'] = df['instrument'].apply(
    lambda x: instrument_map[x] if x in instrument_map else x)
df['file_path'] = df['file_path'].apply(
    lambda x: x.replace(
        '/Users/stephencarrow/Documents/DS-GA 3001 Signal Processing and Deep Learning for Audio/',
        '/scratch/swc419/'))
df['instrument_name'] = df['instrument']
df['instrument'] = df['instrument'].astype('category').cat.codes
# df = df[['file_path', 'songId', 'trackId', 'instrument', 'instrument_name','trackVolume']]
df = df[['file_path', 'songId', 'trackId', 'instrument', 'trackVolume']]
df.to_csv('metadata/ds100.csv', index=False)

# df[['instrument', 'instrument_name']].drop_duplicates()
