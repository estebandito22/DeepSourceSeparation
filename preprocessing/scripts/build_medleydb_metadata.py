import numpy as np
import pandas as pd
import librosa
import os
import glob
import uuid
from collections import defaultdict

import medleydb as mdb

data = defaultdict(list)
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

df = pd.DataFrame(data)
df['file_path'] = df['file_path'].apply(
    lambda x: x.replace(
        '/Users/stephencarrow/Documents/DS-GA 3001 Signal Processing and Deep Learning for Audio/',
        '/scratch/swc419/'))
df['instrument_name'] = df['instrument']
top_instruments
top_instruments = df['instrument'].value_counts().to_frame().query("instrument > 30").index.values
df['instrument'] = df['instrument'].apply(lambda x: x if x in top_instruments else 'other')
df['instrument'] = df['instrument'].astype('category').cat.codes
# df = df[['file_path', 'songId', 'trackId', 'instrument', 'instrument_name','trackVolume']]
df = df[['file_path', 'songId', 'trackId', 'instrument', 'trackVolume']]
df.to_csv('metadata/medleydbV1.csv', index=False)

df['instrument'].value_counts()
df[['instrument', 'instrument_name']].drop_duplicates().sort_values('instrument')
