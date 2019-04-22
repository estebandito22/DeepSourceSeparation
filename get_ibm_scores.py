
import json
from argparse import ArgumentParser

import musdb
import medleydb as mdb

from dss.utils.utils import IBM, IRM
from dss.utils.utils import medians, means
from dss.utils.utils import MedleyTrack

def score(dataset='musdb18'):

    if dataset == 'musdb18':
        # initiate musdb
        mus = musdb.DB(root_dir="/data/musdb18")
        tracks = mus.load_mus_tracks(
            tracknames=['Music Delta - Gospel',
                        'Music Delta - Reggae',
                        'Music Delta - Rock',
                        'ANiMAL - Rockshow',
                        "Actions - Devil's Words",
                        'Johnny Lokke - Whisper To A Scream',
                        'Auctioneer - Our Future Faces',
                        'St Vitus - Word Gets Around',
                        'Strand Of Oaks - Spacestation',
                        'Sweet Lights - You Let Me Down'])

    elif dataset == 'medleydb':
        with open('preprocessing/resources/index_simple_test00.json', 'r') as f:
            test_songs = json.load(f)
        dataset_subset = mdb.load_multitracks(test_songs['id'])
        tracks = []
        for mtrack in dataset_subset:
            tracks.append(MedleyTrack(mtrack))

    # compute IBM stats
    datas = []
    for track in tracks:
        estimates, data = IRM(track, alpha=2)
        datas.append(data)

    return datas


mdb_datas_irm2 = score('medleydb')

print(medians(mdb_datas_irm2))

print(means(mdb_datas_irm2))


# IBM alpha = 2 Medians
# defaultdict(<class 'list'>, {'electric guitar': 6.776717872727273, 'drum set': 4.152351562318841, 'electric bass': 2.778076601751208, 'fx/processed sound': 4.236674929577465, 'male singer': 4.050186699513565, 'other': 6.258032705314009, 'vocalists': 1.9386947872340428, 'piano': 2.338359277284656, 'female singer': 6.6081438})

# IBM alpha = 2 Means
# defaultdict(<class 'list'>, {'electric guitar': 7.369379012229083, 'drum set': 4.82147087658701, 'electric bass': 3.7438955977384296, 'fx/processed sound': 6.459517430961531, 'male singer': 4.258387550817038, 'other': 6.966516648894065, 'vocalists': 2.6517566726975694, 'piano': 4.447232560382003, 'female singer': 7.3373767889809445})

# IRM alpha = 2 Medians
# defaultdict(<class 'list'>, {'electric guitar': 8.510638523903124, 'drum set': 6.421348285714286, 'electric bass': 4.543580776315789, 'fx/processed sound': 5.812093070422535, 'male singer': 5.480917264764389, 'other': 7.9870561352657, 'vocalists': 3.3173169454545453, 'piano': 3.877583443438914, 'female singer': 7.757781014084506})

# IRM alpha = 2 Means
# defaultdict(<class 'list'>, {'electric guitar': 9.031563666713803, 'drum set': 6.454558745479754, 'electric bass': 5.274028731046034, 'fx/processed sound': 8.163997539156671, 'male singer': 5.7074985172917865, 'other': 8.58020762418456, 'vocalists': 4.077884825784395, 'piano': 6.045519726889455, 'female singer': 8.441735065865783})
