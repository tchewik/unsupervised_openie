import _pickle as pickle
import glob
import multiprocessing
import os

import fire
import numpy as np
import pandas as pd
from iteration_utilities import unique_everseen
from tqdm import tqdm

from unidecode import unidecode

INPUT_DIR = '../../data/embedded/'
OUTPUT_DIR = '../../data/final_features/'

MAX_PHRASE_LEN = 4
PREPOSITIONS = "above across after against along among around at away before behind below beneath beside between by " \
               "down during for from in front inside into near next of off on onto out outside over through till to toward " \
               "under underneath until up"
PREPOSITIONS = PREPOSITIONS.split()
DEPREC_RELS = []


def _extract_plain_features(document):
    def _extract(triplets):

        def get_postags_sequence(pos_tags: str):
            postag_types = ['JJ', 'CD', 'VBD', '', 'RB', 'VBN', 'PRP', 'IN', 'VBP', 'TO', 'NNP', 'VB',
                            'VBZ', 'VBG', 'POS', 'NNS', 'NN', 'MD']

            pos = [[int(seq == postag) for postag in postag_types] for seq in pos_tags.split('_')]
            # result = np.zeros((MAX_PHRASE_LEN, len(postag_types)))

            # if pos:
            #     result[:len(pos)] = pos
            result = np.max(pos, axis=0)

            return result

        def get_ner_occurrences(mentions):
            _ner_kinds = ['TITLE', 'COUNTRY', 'DATE', 'PERSON', 'ORGANIZATION', 'MISC',
                          'LOCATION', 'NUMBER', 'CAUSE_OF_DEATH', 'NATIONALITY', 'ORDINAL',
                          'DURATION', 'CRIMINAL_CHARGE', 'CITY', 'RELIGION',
                          'STATE_OR_PROVINCE', 'IDEOLOGY', 'SET', 'URL', 'PERCENT', 'TIME',
                          'MONEY', 'HANDLE']

            mentions = [[int(_ner_kind == mention) for _ner_kind in _ner_kinds] for mention in mentions.split('_')][
                       :MAX_PHRASE_LEN]

            result = np.max(mentions, axis=0)
            # result = np.zeros((MAX_PHRASE_LEN, len(_ner_kinds)))
            #
            # if mentions:
            #     result[:len(mentions)] = mentions

            return result

        def get_prep_sequence(words):
            _prep_kinds = ['above', 'across', 'after', 'against', 'along', 'among', 'around', 'at', 'away',
                           'before', 'behind', 'below', 'beneath', 'beside', 'between', 'by',
                           'down', 'during', 'for', 'from', 'in', 'front', 'inside', 'into',
                           'near', 'next', 'of', 'off', 'on', 'onto', 'out', 'outside', 'over',
                           'through', 'till', 'to', 'toward', 'under', 'underneath', 'until', 'up']

            words = unidecode(words).lower().split(' ')
            result = [int(prep in words) for prep in _prep_kinds]

            return result

        def remove_repetition(words):
            if words[:len(words) // 2] == words[len(words) // 2:]:
                return words[:len(words) // 2]
            return words

        def get_tokens(phrase, delimiter=' '):
            return phrase.lower().split(delimiter)

        #         deprec_rels = {'in',
        #                        #                            'is', 'was',
        #                        'of', "'s", 'to', 'for', 'by', 'with', 'also', 'as of',
        #                        #                            'had',
        #                        'said', 'said in', 'felt', 'on', 'gave', 'saw', 'found', 'did',
        #                        'at', 'as', 'e', 'as', 'de', 'mo', '’s', 'v', 'yr', 'al',
        #                        "'", 'na', 'v.', "d'", 'et', 'mp', 'di', 'y',
        #                        'ne', 'c.', 'be', 'ao', 'mi', 'im', 'h',
        #                        'has', 'between', 'are', 'returned', 'began', 'became',
        #                        'along', 'doors as', 'subsequently terrytoons in',
        #                        }

        filtered_triplets = filter(lambda obj: obj['FEAT_form_rel'].lower().strip() not in DEPREC_RELS,
                                   triplets)
        filtered_triplets = filter(
            lambda obj: len(obj['FEAT_form_subj']) > 2 and len(obj['FEAT_form_obj']) > 2 and len(
                obj['FEAT_form_rel']) > 2,
            filtered_triplets)
        filtered_triplets = filter(lambda obj: obj['FEAT_pair_ntype'] != 'O_O', filtered_triplets)
        filtered_triplets = filter(lambda obj: len(obj['_relation_span']) <= MAX_PHRASE_LEN, filtered_triplets)
        filtered_triplets = filter(lambda obj: len(obj['_subject_span']) <= MAX_PHRASE_LEN, filtered_triplets)
        filtered_triplets = filter(lambda obj: len(obj['_object_span']) <= MAX_PHRASE_LEN, filtered_triplets)
        filtered_triplets = list(filtered_triplets)

        subjects, relations, objects, dep_path = [], [], [], []

        for triplet in filtered_triplets:
            _subject = {
                'tokens': get_tokens(triplet['FEAT_form_subj']),
                'dist_to_rel': triplet['_relation_span'][0] - triplet['_subject_span'][0],
                'pos': get_postags_sequence(triplet['FEAT_pos_subj']),
                'ner': get_ner_occurrences(triplet['FEAT_subj_type']),
                # 'emb': triplet['EMB_subj'],
            }

            _relation = {
                'tokens': get_tokens(triplet['FEAT_form_rel']),
                'lemmas': get_tokens(triplet['FEAT_trigger'], delimiter='|'),
                'pos': get_postags_sequence(triplet['FEAT_pos_rel']),
                'subject_at_left': int(triplet['_subject_span'][0] < triplet['_object_span'][0]),
                'prep': get_prep_sequence(triplet['FEAT_form_rel']),
                # 'emb': triplet['EMB_rel'],
            }

            _object = {
                'tokens': get_tokens(triplet['FEAT_form_obj']),
                'dist_to_rel': triplet['_relation_span'][0] - triplet['_object_span'][0],
                'pos': get_postags_sequence(triplet['FEAT_pos_obj']),
                'ner': get_ner_occurrences(triplet['FEAT_obj_ntype']),
                # 'emb': triplet['EMB_obj'],
            }

            _dependency_path = triplet['_dep_path']
            subjects.append(_subject)
            relations.append(_relation)
            objects.append(_object)
            dep_path.append(_dependency_path)

        return subjects, relations, objects, dep_path

    _subject, _relation, _object, _dep_path = _extract(document[1])
    return _subject, _relation, _object, _dep_path


def _mark_ner_object(row):
    return row['relation'] + (row['DATE_obj'] == 1) * ' date' \
           + (row['LOCATION_obj'] == 1) * ' location'


def _extract_features(document):
    # def _embed_arg(row):
    #     result = []
    #     result.append(_embed(np.zeros((3, word2vec_vector_length)), row['lemmas']))
    #
    #     return result

    features = {}
    features['subject'], features['relation'], features['object'], features['dep_path'] = _extract_plain_features(
        document)
    features['embedding'] = [document[2] for _ in range(len(features['subject']))]

    return pd.DataFrame(features)


def remove_repetitions(annot):
    for i in range(len(annot)):
        for j in range(len(annot[i])):
            annot[i][j]['openie'] = list(unique_everseen(annot[i][j]['openie']))
    return annot


class FeaturesProcessor:

    def __init__(self, num_proc=1):
        self.pool = multiprocessing.Pool(processes=num_proc)

    def __call__(self, data):
        """
        data: list of lists of dicts of keys like "_docid", "_object_span", etc.
        """

        def mark_garbage(row):
            """ Remove from the set some uninformative relations as well as triplets which do not contain
                any noun in the object or subject
            """

            def is_relation_deprecated():
                return row._relation.isdigit() or row._relation in DEPREC_RELS

            def is_postag_undefined():
                return np.all(row['subject']['postag'] == np.zeros((MAX_PHRASE_LEN, 18))) or np.all(
                    row['object']['postag'] == np.zeros((MAX_PHRASE_LEN, 18))) or np.all(
                    row['relation']['postag'] == np.zeros((MAX_PHRASE_LEN, 18)))

            return is_relation_deprecated()  # or is_postag_undefined()

        features = self.pool.map(_extract_features, data)  # each sublist represents a document in a dataset
        if len(features):
            features = pd.concat(features)
            features['_subject'] = features['subject'].map(lambda row: ' '.join(row['tokens']))
            features['_relation'] = features['relation'].map(lambda row: ' '.join(row['tokens']))
            features['_object'] = features['object'].map(lambda row: ' '.join(row['tokens']))
            features['garbage'] = features.apply(lambda row: mark_garbage(row), axis=1)
            features = features[features.garbage == False]
            features = features.drop(columns=["garbage"])
            return features

        return -1


def extract_plain_features(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR, num_threads=6):
    extr = FeaturesProcessor(num_threads)
    dataset = []
    step = 1
    dump_step = 1000

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    for filename in tqdm(glob.glob(os.path.join(input_dir, '*.pkl'))):
        data = pickle.load(open(filename, 'rb'))
        result = extr(data)
        if type(result) != int:
            dataset.append(result)

        if step % dump_step == 0:
            dataset = pd.concat(dataset)
            dataset = dataset.drop_duplicates(['_subject', '_object']).reset_index(drop=True)
            dataset.to_feather(os.path.join(output_dir, f'{step}.fth'))
            dataset = []

        step += 1

    dataset = pd.concat(dataset)
    dataset = dataset.drop_duplicates(['_subject', '_object']).reset_index(drop=True)
    dataset.to_feather(os.path.join(output_dir, f'{step}.fth'))


if __name__ == '__main__':
    fire.Fire(extract_plain_features)
