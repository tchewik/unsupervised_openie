import os
from pathlib import Path

import numpy as np
import pandas as pd
import stanza
import wget
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from pycorenlp import StanfordCoreNLP
from tqdm import tqdm

tqdm.pandas()


class TripletsParser:
    def __init__(self, *args, **kwargs):
        self.verbose = kwargs.get('verbose')
        self.nlp = None

        self.W2V_MODEL_PATH = os.path.dirname('models/')
        self.W2V_MODEL_NAME = 'wiki-news-300d-1M.vec.zip'

        self.word2vec_model = None
        self.word2vec_vector_length = None

        self.postag_tagtypes = {
            'XPOS': ['JJ', 'CD', 'VBD', '', 'RB', 'VBN', 'PRP', 'IN', 'VBP', 'TO', 'NNP', 'VB',
                     'VBZ', 'VBG', 'POS', 'NNS', 'NN', 'MD'],
            'UPOS': ['ADJ', 'ADP', 'PUNCT', 'ADV', 'AUX', 'SYM', 'INTJ', 'CCONJ', 'X',
                     'NOUN', 'DET', 'PROPN', 'NUM', 'VERB', 'PART', 'PRON', 'SCONJ'],
        }

        self.ner_tagtypes = {
            'ontonotes': ['PER', 'ORG', 'MISC', 'PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT',
                          'WORK_OF_ART', 'LAW', 'LANGUAGE', 'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY',
                          'ORDINAL', 'CARDINAL'],
            'corenlp': ['TITLE', 'COUNTRY', 'DATE', 'PERSON', 'ORGANIZATION', 'MISC',
                        'LOCATION', 'NUMBER', 'CAUSE_OF_DEATH', 'NATIONALITY', 'ORDINAL',
                        'DURATION', 'CRIMINAL_CHARGE', 'CITY', 'RELIGION',
                        'STATE_OR_PROVINCE', 'IDEOLOGY', 'SET', 'URL', 'PERCENT', 'TIME',
                        'MONEY', 'HANDLE'],
        }

        self._prep_kinds = ['above', 'across', 'after', 'against', 'along', 'among', 'around', 'at', 'away',
                            'before', 'behind', 'below', 'beneath', 'beside', 'between', 'by',
                            'down', 'during', 'for', 'from', 'in', 'front', 'inside', 'into',
                            'near', 'next', 'of', 'off', 'on', 'onto', 'out', 'outside', 'over',
                            'through', 'till', 'to', 'toward', 'under', 'underneath', 'until', 'up']

    def print(self, text, *args, **kwargs):
        if self.verbose:
            print(text, *args, **kwargs)

    def annotate(self, df):
        assert 'subject_decoded' in df.keys() and 'property_decoded' in df.keys() and 'object_decoded' in df.keys()

        df['subject_annot'] = df.subject_decoded.progress_map(self.nlp)
        df['property_annot'] = df.property_decoded.progress_map(self.nlp)
        df['object_annot'] = df.object_decoded.progress_map(self.nlp)

        return df

    def _initialize_w2v(self):
        if not Path(self.W2V_MODEL_PATH).is_dir():
            self.print(f'Embedder was not found. '
                       f'Creating directory at {self.W2V_MODEL_PATH}'
                       f'for saving word2vec pre-trained model')
            os.makedirs(self.W2V_MODEL_PATH)

        w2v_archive = os.path.join(self.W2V_MODEL_PATH, self.W2V_MODEL_NAME)
        if not Path(w2v_archive).is_file():
            url = f'https://dl.fbaipublicfiles.com/fasttext/vectors-english/{self.W2V_MODEL_NAME}'
            self.print(f'Downloading word2vec pre-trained model to {w2v_archive}')
            wget.download(url, w2v_archive)

        if self.W2V_MODEL_NAME[-4:] in ['.vec', '.bin']:
            self.word2vec_model = KeyedVectors.load_word2vec_format(w2v_archive,
                                                                    binary=self.W2V_MODEL_NAME[-4:] == '.bin')
        elif self.W2V_MODEL_NAME[-4:] == '.zip':
            self.word2vec_model = KeyedVectors.load_word2vec_format(w2v_archive[:-4],
                                                                    binary=self.W2V_MODEL_NAME[-4:] == '.bin')
        elif self.W2V_MODEL_NAME[-7:] == '.bin.gz':
            self.word2vec_model = KeyedVectors.load_word2vec_format(w2v_archive, binary=True)

        else:
            self.word2vec_model = Word2Vec.load(w2v_archive)

        self.word2vec_vector_length = len(self.word2vec_model.wv.get_vector('tree'))

    def _extract_matrix(self, row):
        _matrix = np.concatenate([row[0]['ner'], row[0]['postag'], row[0]['w2v']], axis=1)
        return _matrix

    def _extract_one_matrix(self, row):
        _matrix = np.concatenate([self._extract_matrix(row['subject']),
                                  self._extract_matrix(row['relation']),
                                  self._extract_matrix(row['object'])], axis=0)
        return _matrix

    def _extract_matrices(self, row):

        object_matr = np.stack(self._extract_matrix(row['object']))
        subject_matr = np.stack(self._extract_matrix(row['subject']))
        relation_matr = np.stack(self._extract_matrix(row['relation']))

        return object_matr, subject_matr, relation_matr

    def _extract_plain_features(self, document):
        return [], [], []

    def _extract_features(self, document):
        features = dict()
        features['subject'], features['relation'], features['object'] = self._extract_plain_features(document)

        return features

    def extract_features(self, df):
        if not self.word2vec_model:
            self._initialize_w2v()

        res = df.progress_apply(self._extract_features, axis=1)
        return res.progress_apply(self._extract_matrices)


class TripletsParserStanza(TripletsParser):
    def __init__(self, *args, **kwargs):
        TripletsParser.__init__(self, *args, **kwargs)
        self.nlp = stanza.Pipeline(lang='en', processors='tokenize,lemma,mwt,pos,ner')

    def _extract_plain_features(self, row):
        def _extract(document):

            def get_postags_sequence(sequence, tagtype='UPOS'):

                columns = self.postag_tagtypes[tagtype]

                sequence = sequence[:3]

                result = np.zeros((3, len(columns)))
                sequence = [[int(column == postag) for column in columns] for postag in sequence]

                if sequence:
                    result[:len(sequence)] = sequence

                return result

            def get_ner_occurrences(ner_annot, tagtype='ontonotes'):

                _ner_kinds = self.ner_tagtypes[tagtype]

                ner_annot = ner_annot[:3]

                mentions = [entity.type for entity in ner_annot]
                mentions = [[int(_ner_kind == mention) for _ner_kind in _ner_kinds] for mention in mentions][:3]
                result = np.zeros((3, len(_ner_kinds)))

                if mentions:
                    result[:len(mentions)] = mentions

                return result

            def _embed(placeholder, words):
                for j in range(len(words)):
                    if j == len(placeholder):
                        break

                    word = words[j]
                    if word and word in self.word2vec_model:
                        placeholder[j, :] = self.word2vec_model[word]

                return placeholder

            _object = {
                'tokens': [token.text for token in document.object_annot.sentences[0].tokens],
                'lemmas': [token.lemma for token in document.object_annot.sentences[0].words],
                'ner': get_ner_occurrences(document.object_annot.ents),
                'postag': get_postags_sequence(
                    [token.upos for token in document.object_annot.sentences[0].words]),
            }
            _object.update({
                'w2v': _embed(np.zeros((3, self.word2vec_vector_length)), _object['lemmas']),
            })
            _relation = {
                'tokens': [token.text for token in document.property_annot.sentences[0].tokens],
                'lemmas': [token.lemma for token in document.property_annot.sentences[0].words],
                'ner': get_ner_occurrences(document.property_annot.ents),
                'postag': get_postags_sequence(
                    [token.upos for token in document.property_annot.sentences[0].words]),
            }
            _relation.update({
                'w2v': _embed(np.zeros((3, self.word2vec_vector_length)), _relation['lemmas']),
            })
            _subject = {
                'tokens': [token.text for token in document.subject_annot.sentences[0].tokens],
                'lemmas': [token.lemma for token in document.subject_annot.sentences[0].words],
                'ner': get_ner_occurrences(document.subject_annot.ents),
                'postag': get_postags_sequence(
                    [token.upos for token in document.subject_annot.sentences[0].words]),
            }
            _subject.update({
                'w2v': _embed(np.zeros((3, self.word2vec_vector_length)), _subject['lemmas']),
            })

            subjects, relations, objects, dep_path = [], [], [], []
            subjects.append(_subject)
            relations.append(_relation)
            objects.append(_object)

            return subjects, relations, objects

        return _extract(row)


class TripletsParserCoreNLP(TripletsParser):
    def __init__(self, address, *args, **kwargs):
        TripletsParser.__init__(self, *args, **kwargs)

        if 'http' not in address:
            address = 'https://' + address

        self.nlp = StanfordCoreNLP(address)
        self.nlp_delimiter = 'SPLIT'
        self.nlp_properties = {
            'annotators': 'tokenize,ssplit,tokenize,ssplit,pos,depparse,natlog,openie,ner',
            'ssplit.boundaryTokenRegex': r'(SPLIT)',
            'outputFormat': 'json'
        }

    def annotate_by_chunks(self, chunks):

        self.print("Annotate chunks with CoreNLP...\t", end="", flush=True)
        result = [self.nlp.annotate(chunk, properties=self.nlp_properties)['sentences'] for chunk in chunks]
        self.print("DONE.")

        return result

    def make_chunks_from_texts(self, column):

        tmp = column + ' ' + self.nlp_delimiter + ' '
        text_to_annotate = ['']

        for question in tmp.values:

            if len(text_to_annotate[-1]) + len(question) < 10000:
                text_to_annotate[-1] += question
            else:
                text_to_annotate.append(question)

        self.print(f"Amount of chunks: {len(text_to_annotate)}")

        return text_to_annotate

    def parse_chunks(self, annotations):
        annot = []

        for index, chunk in enumerate(annotations):
            for sentence in annotations[index]:
                annot.append(sentence)

        return pd.Series(annot)

    def annotate_by_chunks_pipeline(self, column):
        chunks = self.make_chunks_from_texts(column)
        result = self.annotate_by_chunks(chunks)
        return self.parse_chunks(result)

    def _extract(self, document):

        def get_postags_sequence(sequence, tagtype='XPOS'):

            columns = self.postag_tagtypes[tagtype]

            sequence = sequence[:3]

            result = np.zeros((3, len(columns)))
            sequence = [[int(column == postag) for column in columns] for postag in sequence]

            if sequence:
                result[:len(sequence)] = sequence

            return result

        def get_ner_occurrences(ner_annot, tagtype='corenlp'):

            _ner_kinds = self.ner_tagtypes[tagtype]

            ner_annot = ner_annot[:3]

            mentions = [entity['ner'] for entity in ner_annot]
            mentions = [[int(_ner_kind == mention) for _ner_kind in _ner_kinds] for mention in mentions][:3]
            result = np.zeros((3, len(_ner_kinds)))

            if mentions:
                result[:len(mentions)] = mentions

            return result

        def get_prep_occurrences(tokens):

            words = [token['word'] for token in tokens]
            mentions = [int(prep in words) for prep in self._prep_kinds]
            result = np.zeros(len(self._prep_kinds))

            if mentions:
                result[:len(mentions)] = mentions

            return result

        def _embed(placeholder, words):
            for j in range(len(words)):
                if j == len(placeholder):
                    break

                word = words[j]
                if word and word in self.word2vec_model:
                    placeholder[j, :] = self.word2vec_model[word]

            return placeholder

        _object = {
            'tokens': [token['word'] for token in document.object_annot['tokens']],
            'lemmas': [token['lemma'].lower() for token in document.object_annot['tokens']],
            'ner': get_ner_occurrences(document.object_annot['entitymentions']),
            'postag': get_postags_sequence(
                [token['pos'] for token in document.object_annot['tokens']]),
        }
        _object.update({
            'w2v': _embed(np.zeros((3, self.word2vec_vector_length)), _object['lemmas']),
        })
        _relation = {
            'tokens': [token['word'] for token in document.property_annot['tokens']],
            'lemmas': [token['lemma'].lower() for token in document.property_annot['tokens']],
            'ner': get_ner_occurrences(document.property_annot['entitymentions']),
            'postag': get_postags_sequence(
                [token['pos'] for token in document.property_annot['tokens']]),
            'prep': get_prep_occurrences(document.property_annot['tokens']),
        }
        _relation.update({
            'w2v': _embed(np.zeros((3, self.word2vec_vector_length)), _relation['lemmas']),
        })
        _subject = {
            'tokens': [token['word'] for token in document.subject_annot['tokens']],
            'lemmas': [token['lemma'].lower() for token in document.subject_annot['tokens']],
            'ner': get_ner_occurrences(document.subject_annot['entitymentions']),
            'postag': get_postags_sequence(
                [token['pos'] for token in document.subject_annot['tokens']]),
        }
        _subject.update({
            'w2v': _embed(np.zeros((3, self.word2vec_vector_length)), _subject['lemmas']),
        })

        subjects, relations, objects, dep_path = [], [], [], []
        subjects.append(_subject)
        relations.append(_relation)
        objects.append(_object)

        return subjects, relations, objects

    def _extract_matrix(self, row, predicate=False):
        _matrix = np.concatenate([row[0]['ner'], row[0]['postag']], axis=1)
        if predicate:
            _matrix = np.concatenate([_matrix, row[0]['w2v'], [row[0]['prep'], ] * 3], axis=1)
        return _matrix

    def _extract_one_matrix(self, row):
        _matrix = np.concatenate([self._extract_matrix(row['subject']),
                                  self._extract_matrix(row['relation'], predicate=True),
                                  self._extract_matrix(row['object'])], axis=0)
        return _matrix

    def _extract_matrices(self, row):
        object_matr = np.stack(self._extract_matrix(row['object']))
        subject_matr = np.stack(self._extract_matrix(row['relation'], predicate=True))
        relation_matr = np.stack(self._extract_matrix(row['object']))

        return object_matr, subject_matr, relation_matr

    def _extract_features(self, document):
        features = dict()
        features['subject'], features['relation'], features['object'] = self._extract(document)

        return features

    def annotate(self, df):
        assert 'subject_decoded' in df.keys() and 'property_decoded' in df.keys() and 'object_decoded' in df.keys()

        df['subject_annot'] = self.annotate_by_chunks_pipeline(df.subject_decoded)
        df['property_annot'] = self.annotate_by_chunks_pipeline(df.property_decoded)
        df['object_annot'] = self.annotate_by_chunks_pipeline(df.object_decoded)

        return df
