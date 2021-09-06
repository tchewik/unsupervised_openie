import glob
import os

import fire
import pandas as pd
from tqdm import tqdm
from unidecode import unidecode

PATH_FINAL_FEATURES = '../../data/matrices/*.fth'
PATH_SQ = '../../data/simplequestions/'
OUT_DIR = '../../data/simplequestions/openie_matched/'
OUT_MATCHED_FILE = 'matched.csv'


def match_data(verbose=True):
    def verbose_print(text='', *args):
        if verbose:
            print(text, *args)

    #########################
    verbose_print('Load openie final features...')

    openie = []
    for file in tqdm(glob.glob(PATH_FINAL_FEATURES)):
        openie.append(pd.read_feather(file)[['_subject', '_relation', '_object']])

    openie = pd.concat(openie)
    openie['subject'] = openie._subject.map(lambda row: unidecode(row.lower().strip()))
    openie['object'] = openie._object.map(lambda row: unidecode(row.lower().strip()))
    openie['relation'] = openie._relation.map(lambda row: unidecode(row.lower().strip()))
    openie = openie.drop_duplicates(['subject', 'object'])
    verbose_print('openie.shape =', openie.shape)

    #########################
    verbose_print('Throw out the matched objects, save new train/valid/test sets...')
    data = {}
    intersections = {}
    matched = []

    for part in ["train", "valid", "test"]:
        path = os.path.join(PATH_SQ, f'annotated_wd_data_{part}_answerable_decoded.csv')
        data[part] = pd.read_csv(path)  # .drop(columns=["Unnamed: 0", "Unnamed: 0.1"])

        simplequestions = data[part][['subject_decoded', 'object_decoded', 'property_decoded', 'question']]
        simplequestions['subject'] = simplequestions.subject_decoded.map(lambda row: unidecode(row.lower().strip()))
        simplequestions['object'] = simplequestions.object_decoded.map(lambda row: unidecode(row.lower().strip()))
        simplequestions = simplequestions.drop_duplicates(['subject', 'object'])
        simplequestions.property_decoded = simplequestions.property_decoded.map(lambda row: row.strip())
        verbose_print(f'sq {path} shape:', simplequestions.shape)

        intersection = pd.merge(simplequestions,
                                openie,
                                how='left',
                                left_on=['subject', 'object'], right_on=['subject', 'object'])
        verbose_print('intersection property_decoded value counts:', len(intersection.property_decoded.value_counts()))

        simplequestions_rev = simplequestions[:].rename(columns={'subject': 'object',
                                                                 'object': 'subject'})
        simplequestions_rev.property_decoded += ' [REV]'
        intersection2 = pd.merge(simplequestions_rev, openie, how='left',
                                 left_on=['subject', 'object'], right_on=['subject', 'object'])
        intersection['relation_inv'] = intersection2.relation

        intersections[part] = intersection[intersection._subject.notna() & (intersection.relation.notna() | intersection.relation_inv.notna())]#.index
        verbose_print('intersection length:', len(intersections[part]))
        matched.append(intersections[part])
        dd = data[part].drop(intersections[part].index)
        verbose_print(f'new length of {part.upper()}:', dd.shape[0])

        if not os.path.isdir(OUT_DIR):
            os.mkdir(OUT_DIR)
        dd.to_csv(os.path.join(OUT_DIR, part + '.csv'), index=None)

        verbose_print()

    #########################
    verbose_print(f'Save matched questions in {os.path.join(OUT_DIR, OUT_MATCHED_FILE)}...')
    matched = pd.concat(matched).drop(columns=['_subject', '_object', '_relation', 'subject', 'object', 'relation'])

    # intersection = pd.concat(intersection)
    verbose_print('Matched questions are of shape:', matched.shape)
    valid_properties = list(matched.property_decoded.unique())
    verbose_print(f'There are {len(valid_properties)} unique wikidata-properties.')
    matched.to_csv(os.path.join(OUT_DIR, OUT_MATCHED_FILE))

    #########################
    verbose_print(f'Filter SimpleQuestions to save only the valid wikidata-properties...')
    new_length, old_length = 0., 0.
    for part in ["train", "valid", "test"]:
        path = os.path.join(OUT_DIR, part + '.csv')
        data[part] = pd.read_csv(path)
        old_length += data[part].shape[ 0]
        data[part]['remove'] = data[part].property_decoded.map(lambda row: row not in valid_properties)
        removed = data[part][data[part].remove == True]
        verbose_print("Examples of removed WikiData properties:", removed.property_decoded.value_counts())
        data[part] = data[part][data[part].remove == False]
        data[part].to_csv(os.path.join(OUT_DIR, part + '.csv'), index=None)
        verbose_print(f'New shape of {part.upper()} is {data[part].shape}.')
        new_length += data[part].shape[0]

    verbose_print(f'The filtered corpus is of {new_length / old_length * 100.}%')
    verbose_print('Done.')


if __name__ == '__main__':
    fire.Fire(match_data)
