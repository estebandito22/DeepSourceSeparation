import musdb
import librosa
import copy
import uuid
from tqdm import tqdm
from collections import defaultdict

import pandas as pd

print("Loading MusDB...")
# mus = musdb.DB(root_dir='/Users/stephencarrow/Documents/DS-GA 3001 Signal Processing and Deep Learning for Audio/DeepSourceSeparation/data/musdb18')
mus = musdb.DB(root_dir='/scratch/swc419/DeepSourceSeparation/data/musdb18')

print("Starting Train...")
train_tracks = mus.load_mus_tracks('train')
data = defaultdict(list)
for i, track in tqdm(enumerate(train_tracks)):
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
df['split'] = 'train'
# df.to_csv('/scratch/swc419/DeepSourceSeparation/metadata/musdb18_train.csv')
# df.to_csv('/Users/stephencarrow/Documents/DS-GA 3001 Signal Processing and Deep Learning for Audio/DeepSourceSeparation/metadata/musdb18_train.csv')

print("Starting Test...")
val_tracks = mus.load_mus_tracks('test')
data = defaultdict(list)
for track in tqdm(val_tracks):
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

df2 = pd.DataFrame(data)
df2['split'] = 'test'

print("Combining Datasets...")
musdb18 = pd.concat([df, df2])
musdb18['instrument'] = musdb18['instrument'].astype('category').cat.codes

print("Printing Datasets...")
musdb18.to_csv('/scratch/swc419/DeepSourceSeparation/metadata/musdb18.csv')


# df.to_csv('/scratch/swc419/DeepSourceSeparation/metadata/musdb18_test.csv')
# df.to_csv('/Users/stephencarrow/Documents/DS-GA 3001 Signal Processing and Deep Learning for Audio/DeepSourceSeparation/metadata/musdb18_test.csv')


# train = pd.read_csv('metadata/musdb18_train_STFT_stereo.csv')
# test = pd.read_csv('metadata/musdb18_test_STFT_stereo.csv')
#
#
# train['instrument'] = train['instrument'].astype('category').cat.codes
# test['instrument'] = test['instrument'].astype('category').cat.codes
#
# train['split'] = 'train'
# test['split'] = 'test'
#
# all = pd.concat([train, test])
#
# all.shape
#
# all.to_csv('metadata/musdb18_STFT_stereo.csv')
#
# train_names = pd.read_csv('metadata/musdb18_train.csv')
# test_names = pd.read_csv('metadata/musdb18_test.csv')
#
#
# train_names['track_name'] = train_names.apply(lambda x: x['file_path'].split('.'+x['instrument']+'.')[0].split('/')[-1], axis=1)
# test_names['track_name'] = test_names.apply(lambda x: x['file_path'].split('.'+x['instrument']+'.')[0].split('/')[-1], axis=1)
#
#
# all_names = pd.concat([train_names, test_names])
# all_names.head()
#
# all['urlId'] =  all_names['track_name']
#
# all.to_csv('metadata/musdb18_STFT_stereo.csv')
