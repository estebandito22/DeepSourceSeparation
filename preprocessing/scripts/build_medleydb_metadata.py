import numpy as np
import pandas as pd
import librosa
import os
import glob
import uuid
import json
from collections import defaultdict

import medleydb as mdb


# get stem info
data = defaultdict(list)
mtrack_generator = mdb.load_all_multitracks()
common_path = '/scratch/swc419/DeepSourceSeparation/data/V1'
for mtrack in mtrack_generator:
    track_id = mtrack.track_id
    track_uuid = str(uuid.uuid4())
    zipped_data = zip(mtrack.stem_instruments,
                      list(mtrack.mixing_coefficients['stft'].values()),
                      mtrack.stem_filepaths())

    for i, (instrument, volume, file_path) in enumerate(zipped_data):
        file_path = file_path.replace('/Users/stephencarrow/medleydb/Audio', common_path)
        # file_path = os.path.join(
        #     common_path, track_id,
        #     track_id + '_STEMS', track_id + '_STEM_{:02d}'.format(i+1) + '.wav')
        data['file_path'].append(file_path)
        data['songId'].append(track_uuid)
        data['trackId'].append(str(uuid.uuid4()))
        data['instrument'].append(instrument)
        data['trackVolume'].append(volume)



# init df and combine instruments
df = pd.DataFrame(data)
# df['file_path'] = df['file_path'].apply(
#     lambda x: x.replace(
#         '/Users/stephencarrow/Documents/DS-GA 3001 Signal Processing and Deep Learning for Audio/',
#         '/home/stephencarrow/'))

df['instrument'] = df['instrument'].apply(lambda x: 'electric guitar' if x.find('electric guitar') > -1 else x)
top_instruments = df['instrument'].value_counts().to_frame().query("instrument > 30").index.values
top_instruments = top_instruments.tolist()
top_instruments.remove('synthesizer')

df['instrument'] = df['instrument'].apply(lambda x: x if x in top_instruments else 'other')
df['instrument'] = df['instrument'].astype('category').cat.codes
# df = df[['file_path', 'songId', 'trackId', 'instrument', 'instrument_name','trackVolume']]
df = df[['file_path', 'songId', 'trackId', 'instrument', 'trackVolume']]

# splits
with open('preprocessing/resources/index_simple_train00.json', 'r') as f:
    train_songs = json.load(f)
with open('preprocessing/resources/index_simple_validate00.json', 'r') as f:
    val_songs = json.load(f)
with open('preprocessing/resources/index_simple_test00.json', 'r') as f:
    test_songs = json.load(f)

df['track_name'] = df['file_path'].apply(lambda x: x.split('V1')[-1].split('/')[1])
df['split'] = df['track_name'].apply(lambda x: 'train' if x in train_songs['id'] else ('val' if x in val_songs['id'] else 'test'))

# save
df.to_csv('metadata/medleydbV1_splits_redo.csv', index=False)

# df['instrument'].value_counts()
# df[['instrument', 'instrument_name']].drop_duplicates().sort_values('instrument')
