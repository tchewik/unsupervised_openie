import _pickle as pickle
import os

import fire
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

BASIC_FEATURES_FILE = '../../data/basic_features.pkl'
OUTPUT_PATH = '../../data/embedded/'


def embed(model_name='paraphrase-distilroberta-base-v2', verbose=True):
    def verbose_print(text='', *args):
        if verbose:
            print(text, *args)

    dataset = pickle.load(open(BASIC_FEATURES_FILE, 'rb'))
    sentences = [s[0] for s in dataset]
    verbose_print('Found', len(sentences), 'sentences.')

    model = SentenceTransformer(model_name)
    step = 100
    i = 0
    start = step * i

    if not os.path.isdir(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)

    with tqdm(total=len(sentences)) as pbar:
        while start < len(sentences):
            filename = os.path.join(OUTPUT_PATH, 'annot_{i:04.0f}.pkl')
            if True:  # not os.path.isfile(filename):
                embeddings = model.encode(sentences[start: min(start + step, len(sentences))])
                _annot = dataset[start: min(start + step, len(sentences))]
                for idx, sentemb in enumerate(embeddings):
                    _annot[idx].append(sentemb)

                pickle.dump(_annot, open(filename, 'wb'))

            start += step
            i += 1
            pbar.update(step)


if __name__ == '__main__':
    fire.Fire(embed)
