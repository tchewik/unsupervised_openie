import glob
import logging
import os

import fire
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import plot_model
from tensorflow.python.client import device_lib
from tensorflow.python.keras import backend as K
from tqdm import tqdm
from unidecode import unidecode

import deep_clustering
from autoencoder_models import mish
from autoencoder_models import noised_ae

print(tf.__version__)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

K.set_session(sess)


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


print(get_available_gpus())

DEFAULT_DATA_PATH = '../../data/matrices/'
DEFAULT_CLUSTERS_RANGE = [40, 58, 76, 94, 112]
SECOND_DIMENSION = 1  # originally 3

DEFAULT_MODELS_PATH = '../../models'
AEC_SAVE_H5_PATH = '../../models/pretrain_cae_model.h5'
IDEC_SAVE_DIR = '../../models/idec'
OUTPUT_DATA_DIR = '../../data/clusterized'

log_path = '../../logs'
fileName = 'main.log'
logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")

logger = logging.getLogger()
if logger.hasHandlers():
    logger.handlers.clear()

fileHandler = logging.FileHandler(os.path.join(log_path, fileName))
fileHandler.setFormatter(logFormatter)
logger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)

logger.setLevel(logging.INFO)

for path in [DEFAULT_MODELS_PATH,
             IDEC_SAVE_DIR,
             OUTPUT_DATA_DIR,
             log_path]:
    if not os.path.exists(path):
        os.mkdir(path)


def _train_idec(_subject,
                _object,
                _relation,
                n_clusters,
                data,
                pretrained,
                pretrained_aec_path,
                autoencoder=noised_ae,
                batch_size=256,
                pretrain_batch_size=512,
                pretrain_epochs=100,
                max_iter=300,
                score_threshold=1e-5,
                save_dir=IDEC_SAVE_DIR,
                partial_init=False,
                partial_part='predicate'):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    _directory = os.path.join(save_dir, f'idec_{n_clusters}_partial{str(partial_init)}_part_{partial_part}')
    if not os.path.exists(_directory):
        os.mkdir(_directory)

    if pretrained:
        _autoencoder = keras.models.load_model(pretrained_aec_path, custom_objects={'mish': mish})
    else:
        _autoencoder = lambda input_shape: autoencoder(input_shape)

    input_shape = [_subject.shape[1:], _object.shape[1:], _relation.shape[1:]]
    idec = deep_clustering.IDEC(input_shape,
                                pretrained=pretrained,
                                autoencoder_ctor=_autoencoder,
                                n_clusters=n_clusters,
                                pretrain_epochs=pretrain_epochs,
                                tol=-1.,
                                max_iter=max_iter,
                                partial_init=partial_init,
                                partial_part=partial_part,
                                save_dir=_directory,
                                log_dir=log_path)

    idec.compile(optimizer='adam')
    plot_model(idec._model, to_file='daec_model.png', show_shapes=True)
    idec.fit([_subject, _object, _relation], batch_size=batch_size, pretrain_batch_size=pretrain_batch_size)

    # dump data somewhere
    output_filename = f'emb_idec_clusters_{n_clusters}_partial{str(partial_init)}_{partial_part}.fth'
    print('Save the clusterized data into', os.path.join(OUTPUT_DATA_DIR, output_filename))
    y_pred = idec._y_pred
    dumb_features = data.copy()
    dumb_features['cluster'] = y_pred
    scores = idec.score_examples([_subject, _object, _relation])
    dumb_features['score'] = scores
    dumb_features = dumb_features[dumb_features['score'] > score_threshold]
    dumb_features.reset_index(drop=True).to_feather(
        os.path.join(OUTPUT_DATA_DIR, output_filename))
    print('Done.')
    return idec


def train_models(idec=True,
                 daec=False,
                 dc_kmeans=False,
                 init_aec=False,
                 pretrain_epochs=200,
                 pretrain_batch_size=256,
                 clusters_range=DEFAULT_CLUSTERS_RANGE,
                 partial_init_range=[True, False],
                 partial_part='predicate',
                 data_path=DEFAULT_DATA_PATH):
    if not idec or daec or dc_kmeans:
        print('No deep clustering model is defined. Abort.')
        return

    print('Loading data...')
    data = []
    for file in tqdm(glob.glob(os.path.join(data_path, '*.fth'))):
        data.append(pd.read_feather(file))
    data = pd.concat(data)
    data._subject = data._subject.map(lambda row: unidecode(row.lower().strip()))
    data._object = data._object.map(lambda row: unidecode(row.lower().strip()))
    data = data.drop_duplicates(['_subject', '_object'])

    _object, _subject, _relation = data.object_matr.values, data.subject_matr.values, data.relation_matr.values

    if SECOND_DIMENSION == 1:
        _object = np.array([[row] for row in data.object_matr.values])
        _subject = np.array([[row] for row in data.subject_matr.values])
        _relation = np.array([[row] for row in data.relation_matr.values])
    else:
        _object = np.stack(_object)
        _subject = np.stack(_subject)
        _relation = np.stack(_relation)

    data = data[['_subject', '_relation', '_object']]  # flush data from RAM

    if init_aec:
        if os.path.isfile(AEC_SAVE_H5_PATH):
            os.remove(AEC_SAVE_H5_PATH)

    if not os.path.isfile(AEC_SAVE_H5_PATH):
        print('Pretrain an autoencoder...')
        input_shape = [_subject.shape[1:], _object.shape[1:], _relation.shape[1:]]
        model = noised_ae(input_shape=input_shape)
        model.summary()
        model.compile(optimizer='adam', loss='mse')
        model.fit(x=[_subject, _object, _relation],
                  y=[_subject, _object, _relation], epochs=pretrain_epochs, batch_size=pretrain_batch_size)
        model.save(AEC_SAVE_H5_PATH)
    else:
        print('Found a pretrained autoencoder.')

    if idec:
        print()
        print('Train IDEC...')
        for n_clusters in clusters_range:
            print('n_clusters =', n_clusters)
            for partial_init in partial_init_range:
                print('partial_init =', partial_init)
                if partial_init:
                    for part in ['arguments', 'predicate']:
                        print()
                        print('partial_init part =', part)
                        _train_idec(_subject=_subject,
                                    _object=_object,
                                    _relation=_relation,
                                    data=data,
                                    pretrained=True,
                                    pretrained_aec_path=AEC_SAVE_H5_PATH,
                                    n_clusters=n_clusters,
                                    partial_init=partial_init,
                                    partial_part=part,
                                    )
                        part = ''
                else:
                    print()
                    _train_idec(_subject=_subject,
                                _object=_object,
                                _relation=_relation,
                                data=data,
                                pretrained=True,
                                pretrained_aec_path=AEC_SAVE_H5_PATH,
                                n_clusters=n_clusters,
                                partial_init=partial_init,
                                )

    if daec:
        for n_clusters in clusters_range:
            for partial_init in partial_init_range:
                try:
                    # ToDO;
                    pass
                except:
                    pass

    if dc_kmeans:
        for n_clusters in clusters_range:
            for partial_init in partial_init_range:
                try:
                    # ToDO:
                    pass
                except:
                    pass


if __name__ == '__main__':
    fire.Fire(train_models)
