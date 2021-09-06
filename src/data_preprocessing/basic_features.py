import _pickle as pickle
import glob
import json
import os
import re
import string
import typing
from multiprocessing import Pool

import fire
import networkx as nx
import nltk
from tqdm import tqdm

WIKITEXTS_DIR = '../../data/wiki_texts'
OUTPUT_DIR = '../../data/basic_features'
FINAL_PATH = '../../data/basic_features.pkl'

nltk.download('stopwords')
stopwords_list = nltk.corpus.stopwords.words('english')
_digits = re.compile('\d')

deprec_rels = set()


def extract_tokens(syntax_annot: list, arg1_span: list, arg2_span: list, verbose=False) -> typing.List[int]:
    """
    :param syntax_annot: list, basicDependencies annotation for a sentences
    :param arg1_span: list of type [start (int), end (int)]
    :param arg2_span: list of type [start (int), end (int)]

    :return: indexes of the root tokens for argument phrase #1 and argument phrase #2
    """

    def find_phrase_root(span):
        start, end = [span[0] + 1, span[1] + 1]  # in corenlp dependencies, element #0 is ROOT
        argument_syntax = syntax_annot[start:end]
        idx = 0

        for idx, token in enumerate(argument_syntax):
            if token['governor'] not in range(start, end):
                return start + idx

        return start + idx  # assume the last token in a phrase is the most important

    tok1 = find_phrase_root(arg1_span)
    tok2 = find_phrase_root(arg2_span)

    if verbose:
        print('EXTRACT_TOKENS:', [tok1, tok2])

    return [tok1, tok2]


def _get_pos(tokens: list, span: list, verbose=False) -> str:
    """
    Find the postags for a phrase
    :param tokens: list, tokens annotation from corenlp
    :param span: list of type [start (int), end (int)]

    :return: string including all the PoS tags on span
    """
    result = []
    for pos in [tokens[i]['pos'] for i in range(span[0], span[1])]:
        if pos not in string.punctuation:
            result.append(pos)
    result = '_'.join(result)

    if verbose:
        print('GET_POS:', result)

    return result


def _get_bow_between(tokens: list, arg1_span: list, arg2_span: list, verbose=False) -> str:
    """
    Finds the tokens laying in plain text between two arguments
    :param tokens: list, tokens annotation from corenlp
    :param arg1_span: list of type [start (int), end (int)]
    :param arg2_span: list of type [start (int), end (int)]

    :return: string including all the words between arg#1 and arg#2 in text
    """

    tmp = []
    result = []
    tok_left, tok_right = sorted([arg1_span, arg2_span])
    for word in [tokens[i]['originalText'] for i in range(max(tok_left), min(tok_right))]:
        for pun in string.punctuation:
            word = word.strip(pun)
        if word != '':
            tmp.append(word.lower())

    for word in tmp:
        if not _digits.search(word) and not word[0].isupper():
            result.append(word)

    result = ' '.join(result)

    if verbose:
        print('GET_BOW_BETWEEN:', result)

    return result


def _get_pos_between(tokens: list, arg1_span: list, arg2_span: list, verbose=False) -> str:
    """
    Finds the parts of speech laying in plain text between two arguments
    :param tokens: list, tokens annotation from corenlp
    :param arg1_span: list of type [start (int), end (int)]
    :param arg2_span: list of type [start (int), end (int)]

    :return: string including all the postags between arg#1 and arg#2 in text
    """
    result = []
    tok_left, tok_right = sorted([arg1_span, arg2_span])
    for pos in [tokens[i]['pos'] for i in range(max(tok_left), min(tok_right))]:
        if pos not in string.punctuation:
            result.append(pos)
    result = '_'.join(result)

    if verbose:
        print('GET_POS_BETWEEN:', result)

    return result


def _get_dep_path(dependencies: list, start: int, end: int, verbose=False):
    """
    Finds the shortest dependency path between two tokens in a sentence.
    :param dependencies: list, List of dependencies in Stanford CoreNLP style
    :param start: int, position of the root token in phrase #1
    :param end: int, position of the root token in phrase #2

    :return: list of tokens [start ... end] as they are presented in the shortest dependency path
    """
    edges = []
    deps = {}

    for edge in dependencies:
        edges.append((edge['governor'] - 1, edge['dependent'] - 1))
        deps[(min(edge['governor'] - 1, edge['dependent'] - 1),
              max(edge['governor'] - 1, edge['dependent'] - 1))] = edge

    graph = nx.Graph(edges)
    for i in range(len(edges) // 2):
        try:
            path = nx.shortest_path(graph, source=start, target=end)
            if path:
                break
        except:
            edges = edges[1:-1]
            graph = nx.Graph(edges)
            path = []

    result = [p for p in path]

    if verbose:
        print('GET_DEP_PATH:', result)

    return result


# def _get_shortest_path(dependencies, left_set, right_set):
#     """
#     Finds the shortest dependency path between two sets of tokens in a sentence.
#     """
#     result = [1] * len(dependencies)
#     for a in left_set:
#         for b in right_set:
#             candidate = _get_dep_path(dependencies, a, b)
#             if len(candidate) < len(result):
#                 result = candidate
#
#     print('GET_SHORTEST_PATH:', result)
#     return result


def _get_words_dep(tokens: list, dependency_path: list, verbose=False) -> str:
    """
    Finds tokens forms on the dependency path
    :param tokens: list of tokens in corenlp format
    :param dependency_path: list of token indexes on a dependency path

    :return: string including all the postags on a dependency path
    """
    result = ' '.join([tokens[i]['word'] for i in dependency_path[1:-1]])

    if verbose:
        print('GET_WORDS_DEP:', result)

    return result


def _get_trigger(tokens: list, dependency_path: list, verbose=False) -> str:
    """
    Finds tokens lemmas most likely triggering a relation
    :param tokens: list of tokens in corenlp format
    :param dependency_path: list of token indexes on a dependency path

    :return: string including all the triggering words (normalized) on a dependency path
    """
    result = []
    for word in [tokens[i]['lemma'] for i in dependency_path[1:-1]]:
        if word not in stopwords_list:
            result.append(word)

    result = '|'.join(result)

    if verbose:
        print('GET_TRIGGER:', result)

    return result


def _get_entity_type(tokens, tok_span, verbose=False):
    """
    Finds NE types of the token span
    :param tokens:
    :param tok:
    :return:
    """
    _replace = {
        'PERSON_PERSON': 'PERSON',
        'ORGANIZATION_ORGANIZATION': 'ORGANIZATION',
        'DURATION': 'O',
        'O_O': 'O',
    }

    result = '_'.join([tokens[token].get('ner') for token in range(tok_span[0], tok_span[1])])
    for key, value in _replace.items():
        while key in result:
            result = result.replace(key, value)

    if verbose:
        print('GET_ENTITY_TYPE:', result)

    return result


def process_document(annotation, verbose=False):
    """ annotation: dic is a corenlp annotation for one sentence """
    result = []
    prev_obj, prev_subj = "", ""

    sentence = ' '.join([token['word'] for token in annotation['tokens']])

    for triple in annotation['openie']:
        if triple['object'] and triple['subject'] and not (
                triple['object'] == prev_obj and triple['subject'] == prev_subj):
            tok1, tok2 = extract_tokens(annotation['basicDependencies'],
                                        triple['objectSpan'],
                                        triple['subjectSpan'],
                                        verbose=verbose)

            prev_obj, prev_subj = triple['object'], triple['subject']  # to filter duplicates

            len_arg1 = len(triple['object'].split())
            len_arg2 = len(triple['subject'].split())
            len_rel = len(triple['relation'].split())

            if len_arg1 < 5 and len_arg2 < 5 and len_rel < 5 and not triple['relation'] in deprec_rels:

                surface1 = triple['object']
                surface2 = triple['subject']
                surface_relation = triple['relation']

                pos1 = _get_pos(annotation['tokens'], triple['objectSpan'])
                pos2 = _get_pos(annotation['tokens'], triple['subjectSpan'])
                pos_relation = _get_pos(annotation['tokens'], triple['relationSpan'])

                bow = _get_bow_between(annotation['tokens'], triple['objectSpan'], triple['subjectSpan'],
                                       verbose=verbose)
                pos = _get_pos_between(annotation['tokens'], triple['objectSpan'], triple['subjectSpan'],
                                       verbose=verbose)

                dependency_path = _get_dep_path(annotation['basicDependencies'], tok1, tok2, verbose=verbose)
                trigger = _get_trigger(annotation['tokens'], dependency_path, verbose=verbose)

                ent1 = _get_entity_type(annotation['tokens'], triple['objectSpan'], verbose=verbose)
                ent2 = _get_entity_type(annotation['tokens'], triple['subjectSpan'], verbose=verbose)

                if not ent1 + ent2 == 'OO':
                    path = _get_words_dep(annotation['tokens'], dependency_path, verbose=verbose)

                    result.append({
                        '_docid': annotation['doc_id'],
                        '_object_span': triple['objectSpan'],
                        '_subject_span': triple['subjectSpan'],
                        '_relation_span': triple['relationSpan'],
                        '_dep_path': dependency_path,
                        'FEAT_pos_obj': pos1,
                        'FEAT_pos_subj': pos2,
                        'FEAT_pos_rel': pos_relation,
                        ## Titov features
                        'FEAT_form_obj': surface1,
                        'FEAT_form_subj': surface2,
                        'FEAT_form_rel': surface_relation,
                        'FEAT_trigger': trigger,
                        'FEAT_bow_between': bow,
                        'FEAT_pos_between': pos,
                        'FEAT_pair_ntype': ent1 + '_' + ent2,
                        'FEAT_obj_ntype': ent1,
                        'FEAT_subj_type': ent2,
                        'FEAT_path': path,
                    })

    return sentence, result


def filter_rels(sentence):
    deprecated_rels = 'see,saw,seen,sees,seeing,say,said,says,saying,give,gave,given,gives,giving,take,takes,took,take,taking,feel,felt,feels,feeling'.split(
        ',')
    new_sentence = [sentence[0], []]
    for triplet in sentence[1]:
        if triplet['FEAT_pos_subj'].strip() != 'PRP':
            if triplet['FEAT_pos_obj'].strip() != 'PRP':
                if not triplet['FEAT_form_rel'].lower().strip() in deprecated_rels:
                    new_sentence[1].append(triplet)
    if new_sentence[1]:
        return new_sentence
    else:
        return None


def make_features(num_threads=16, verbose=False):
    def verbose_print(text='', *args):
        if verbose:
            print(text, *args)

    files = glob.glob(os.path.join(WIKITEXTS_DIR, '*.json'))
    files.sort()
    batch_size = 5000
    step = 0

    for i in tqdm(range(len(files) // batch_size + 1)):
        dataset = []
        verbose_print(f'Loading the json data... Batch {step}')
        for idx, file in tqdm(enumerate(files[step * batch_size: min((step + 1) * batch_size, len(files))])):
            annot = json.load(open(file, 'r'))
            for sent in annot:
                sent['doc_id'] = file
            dataset += annot

        verbose_print('Processing...')
        pool = Pool(num_threads)
        result = pool.map(process_document, dataset)
        pickle.dump(result, open(os.path.join(OUTPUT_DIR, f'{step}.pkl'), 'wb'))
        step += 1

    verbose_print(f'Collect {FINAL_PATH}...')
    files = glob.glob(os.path.join(OUTPUT_DIR, '*.pkl'))
    files.sort()

    dataset = []
    for file in tqdm(files):
        annot = pickle.load(open(file, 'rb'))
        dataset += annot

    verbose_print(f'Filter relations and arguments...')
    new_data = []
    for sent in tqdm(dataset):
        sent = filter_rels(sent)
        if sent:
            new_data.append(sent)

    pickle.dump(new_data, open(FINAL_PATH, 'wb'))

    verbose_print('Remove temporary files...')
    for file in tqdm(files):
        os.remove(file)
    os.remove(OUTPUT_DIR)

    verbose_print('Done!')


if __name__ == '__main__':
    fire.Fire(make_features)
