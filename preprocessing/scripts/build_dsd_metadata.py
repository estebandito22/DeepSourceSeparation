import numpy as np
import pandas as pd
import librosa
import os
import glob
import uuid
from collections import defaultdict


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
train = pd.DataFrame(data)
train['file_path'] = train['file_path'].apply(
    lambda x: x.replace(
        '/Users/stephencarrow/Documents/DS-GA 3001 Signal Processing and Deep Learning for Audio/',
        '/scratch/swc419/'))
train['instrument'] = train['instrument'].astype('category').cat.codes
train = train[['file_path', 'songId', 'trackId', 'instrument', 'trackVolume']]
train.to_csv('metadata/ds100_train.csv', index=False)

DATA = os.path.join(CWD, 'data', 'DSD100', 'Sources', 'Test')
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
test = pd.DataFrame(data)
test['file_path'] = test['file_path'].apply(
    lambda x: x.replace(
        '/Users/stephencarrow/Documents/DS-GA 3001 Signal Processing and Deep Learning for Audio/',
        '/scratch/swc419/'))
test['instrument'] = test['instrument'].astype('category').cat.codes
test = test[['file_path', 'songId', 'trackId', 'instrument', 'trackVolume']]
test.to_csv('metadata/ds100_test.csv', index=False)
