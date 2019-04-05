import musdb
import librosa
import copy
import uuid
from collections import defaultdict

import pandas as pd

mus = musdb.DB(root_dir='/Users/stephencarrow/Documents/DS-GA 3001 Signal Processing and Deep Learning for Audio/DeepSourceSeparation/data/musdb18')
# mus = musdb.DB(root_dir='/scratch/swc419/DeepSourceSeparation/data/musdb18')

train_tracks = mus.load_mus_tracks('train')
data = defaultdict(list)
for i, track in enumerate(train_tracks):
    stems = track.sources
    path_list = track.path.split('.')
    songId = str(uuid.uuid4())
    for key in stems.keys():
        write_path = copy.copy(path_list)
        write_path = [write_path[0]] + [key] + ['wav']
        write_path = '.'.join(write_path)
        librosa.output.write_wav(write_path, stems[key].audio, track.rate)
        # build metadata
        data['file_path'] += [write_path]
        data['trackId'] += [str(uuid.uuid4())]
        data['songId'] += [songId]
        data['instrument'] += [key]
        data['trackVolume'] += [stems[key].gain]

df = pd.DataFrame(data)
# df.to_csv('/scratch/swc419/DeepSourceSeparation/metadata/musdb18_train.csv')
df.to_csv('/Users/stephencarrow/Documents/DS-GA 3001 Signal Processing and Deep Learning for Audio/DeepSourceSeparation/metadata/musdb18_train.csv')

val_tracks = mus.load_mus_tracks('test')
data = defaultdict(list)
for track in val_tracks:
    stems = track.sources
    path_list = track.path.split('.stem.')
    songId = str(uuid.uuid4())
    for key in stems.keys():
        write_path = copy.copy(path_list)
        write_path = [write_path[0]] + [key] + ['wav']
        write_path = '.'.join(write_path)
        librosa.output.write_wav(write_path, stems[key].audio, track.rate)
        # build metadata
        data['file_path'] += [write_path]
        data['trackId'] += [str(uuid.uuid4())]
        data['songId'] += [songId]
        data['instrument'] += [key]
        data['trackVolume'] += [stems[key].gain]

df = pd.DataFrame(data)
# df.to_csv('/scratch/swc419/DeepSourceSeparation/metadata/musdb18_test.csv')
df.to_csv('/Users/stephencarrow/Documents/DS-GA 3001 Signal Processing and Deep Learning for Audio/DeepSourceSeparation/metadata/musdb18_test.csv')
