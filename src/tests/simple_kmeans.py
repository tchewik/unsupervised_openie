import glob
import os
import pickle

import fire
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from tqdm import tqdm
from unidecode import unidecode

DEFAULT_DATA_PATH = '../../data/matrices/'
SECOND_DIMENSION = 1
DEFAULT_MODEL_PATH = '../../models/'
DEFAULT_DATA_OUTPUT_PATH = '../../data/clusterized/'


def train_k_means(n):
    print('Loading data...')
    data = []
    for file in tqdm(glob.glob(os.path.join(DEFAULT_DATA_PATH, '*.fth'))):
        data.append(pd.read_feather(file))
    data = pd.concat(data)
    data._subject = data._subject.map(lambda row: unidecode(row.lower().strip()))
    data._object = data._object.map(lambda row: unidecode(row.lower().strip()))
    data = data.drop_duplicates(['_subject', '_object'])

    _object, _subject, _relation = data.object_matr.values, data.subject_matr.values, data.relation_matr.values

    if SECOND_DIMENSION == 1:
        _object = data.object_matr.values
        _subject = data.subject_matr.values
        _relation = data.relation_matr.values
    else:
        _object = np.stack(_object)
        _subject = np.stack(_subject)
        _relation = np.stack(_relation)

    print('SHAPES:')
    print('_subject.shape =', _subject.shape)
    print('_relation.shape =', _relation.shape)
    print('_object.shape =', _object.shape)
    features = np.concatenate([_subject.tolist(), _relation.tolist(), _object.tolist()], axis=-1)
    data = data[['_subject', '_relation', '_object']]  # flush data from RAM

    print('Training k-means...')
    kmeans = KMeans(init='k-means++', n_clusters=n, n_init=10)
    kmeans.fit(features.tolist())
    pickle.dump(kmeans, open(os.path.join(DEFAULT_MODEL_PATH, f'baseline_kmeans_{n}.pkl'), 'wb'))
    data['cluster'] = kmeans.predict(features.tolist())
    data[['_subject', '_relation', '_object', 'cluster']].to_feather(
        os.path.join(DEFAULT_DATA_OUTPUT_PATH, f'baseline_kmeans_{n}.fth'))


if __name__ == '__main__':
    for n in [40, 58, 76, 94, 112]:
        train_k_means(n)
    # fire.Fire(train_k_means)
