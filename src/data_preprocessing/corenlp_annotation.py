import glob
import json
import os
import time

import fire
# import neuralcoref
# import spacy
from pycorenlp import StanfordCoreNLP
from tqdm import tqdm

WIKITEXTS_DIR = '../../data/wiki_texts'
RESTART_TIME = 10  # How long to wait til the container restarts properly


# coref = spacy.load('en_core_web_sm')
# neuralcoref.add_to_pipe(coref)
#
# def resolve_coreference(text):
#     doc = coref(text)
#     return doc._.coref_resolved


def filter_ner(sentence):
    # save only triplets with at least one named entity
    # and does not contain a deprecated relation
    openie = []
    global counter
    deprec_rels = {'in',
                   'of', "'s", 'to', 'for', 'by', 'with', 'also', 'as of',
                   'said', 'said in', 'felt', 'on', 'gave', 'saw', 'found', 'did',
                   'at', 'as', 'as', 'de', 'mo', '’s', 'yr', 'al',
                   'na', 'v.', "d'", 'et', 'mp', 'di',
                   'ne', 'c.', 'be', 'ao', 'mi', 'im',
                   'has', 'between', 'are', 'returned',
                   'along', 'doors as', 'subsequently terrytoons in',
                   }

    for triplet in sentence['openie']:
        if not len(triplet['relation']) < 2:
            if not triplet['relation'] in deprec_rels:
                for entity in sentence['entitymentions']:
                    if entity['text'] in [triplet['subject'], triplet['object']]:
                        openie.append(triplet)
                        continue

    return openie


def parse(hostname, port, start=0):
    nlp = StanfordCoreNLP(f'http://{hostname}:{port}')
    nlp_properties = {
        'annotators': 'tokenize,ssplit,tokenize,pos,ner,depparse,natlog,openie',
        'outputFormat': 'json'
    }

    def corenlp_annotate(text, resolve_coref=False):
        # if resolve_coref:
        #     text = resolve_coreference(text)

        try:
            return nlp.annotate(text, properties=nlp_properties)['sentences']
        except TypeError:
            time.sleep(RESTART_TIME)
            result = nlp.annotate(text, properties=nlp_properties)
            if type(result) == str:
                return None
        return None

    files = glob.glob(os.path.join(WIKITEXTS_DIR, '*.txt'))
    files.sort()

    print("Parse files...")
    for file in tqdm(files[start:]):
        if not os.path.isfile(file.replace('.txt', '.json')):
            with open(file, 'r') as f:
                text = f.read()
                doc_annot = corenlp_annotate(text)
                clean_annot = []
                if doc_annot:
                    for sentence in doc_annot:
                        new_sentence = sentence
                        new_sentence['openie'] = filter_ner(sentence)
                        if new_sentence['openie']:
                            clean_annot.append(new_sentence)

                json.dump(clean_annot, open(file.replace('.txt', '.json'), 'w'))


if __name__ == '__main__':
    fire.Fire(parse)
