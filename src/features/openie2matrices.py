import glob
import os

import fire
import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm

tqdm.pandas()

DEFAULT_INPUT_DIR = '../../data/final_features'
DEFAULT_OUTPUT_DIR = '../../data/matrices'
RANDOM_STATE = 42


def extract_matrix(row, predicate=False):
    if predicate:
        _matrix = np.concatenate([row['pos'], row['prep'], [row['subject_at_left']], row['emb']])
    else:
        _matrix = np.concatenate([row['pos'], row['ner']])
    return _matrix


def make_matrices(input_dir=DEFAULT_INPUT_DIR, output_dir=DEFAULT_OUTPUT_DIR, verbose=True):
    def verbose_print(text='', *args):
        if verbose:
            print(text, *args)

    verbose_print("Load data...")
    data = []
    for file in tqdm(glob.glob(os.path.join(input_dir) + '/*.fth')):
        data.append(pd.read_feather(file))

    data = pd.concat(data).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    data = data.drop_duplicates(['_subject', '_object'])

    # Embeddings of the relation here are the contextual embeddings of the whole sentence
    verbose_print("Load embeddings...")
    data['relation'] = data.apply(lambda row: dict(row.relation, **{'emb': row.embedding}), axis=1)

    verbose_print("Construct feature matrices for (1) Object, (2) Subject, and (3) Relation...")
    data['object_matr'] = data.object.progress_map(extract_matrix)
    data['subject_matr'] = data.subject.progress_map(extract_matrix)
    data['relation_matr'] = data.relation.progress_map(lambda row: extract_matrix(row, predicate=True))

    verbose_print("Dump data...")
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    start = 0
    chunksize = 100000
    for step in tqdm(range(data.shape[0] // chunksize + 1)):
        data[start:min(start + chunksize, data.shape[0])].reset_index().to_feather(
            os.path.join(output_dir, f'{step}.fth'))
        start += chunksize

    if start < data.shape[0]:
        data[start:min(start + chunksize, data.shape[0])].reset_index().to_feather(
            os.path.join(output_dir, f'{step}.fth'))

    verbose_print("Done.")


if __name__ == '__main__':
    fire.Fire(make_matrices)
