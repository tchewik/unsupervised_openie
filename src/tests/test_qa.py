import glob
import os

import fire
import pandas as pd
from unidecode import unidecode
from tqdm import tqdm

from question_classifiers import FastTextClassifier
from train_classifiers import load_matched_data, load_model_predictions, clusters4matched

tqdm.pandas()

DEFAULT_DATA_PATH = '../../data/clusterized'
MATCHED_DATA_FILE = '../../data/simplequestions/openie_matched/matched.csv'


class DummyQuestionParser:
    def get_entities(self, question, subj, obj):
        question = unidecode(question)
        subj, obj = unidecode(subj), unidecode(obj)

        if subj in question:
            return subj
        elif obj in question:
            return obj

        question = question.replace("'", '')
        subj, obj = subj.replace("'", ''), obj.replace("'", '')

        if subj in question:
            return subj
        elif obj in question:
            return obj

        return ''


def compare(obj1, obj2):
    obj1 = obj1.lower().split()
    obj2 = obj2.lower().split()
    return any(set.intersection(set(obj1), set(obj2)))


def evaluate(true_subj, true_obj, prediction):
    for pred in prediction:
        if true_subj in pred and true_obj in pred:
            return True

    return False


def test_qa(strategy='max', output_file='test_results.txt'):
    path_template = os.path.join('../../models/question_classifiers/', strategy, '*.fasttext')
    for classifier in glob.glob(path_template):
        print()
        print('Looking at', classifier)

        predictions_filename = classifier.split('/')[-1]
        predictions_filename = os.path.join(DEFAULT_DATA_PATH, predictions_filename)
        predictions_filename = predictions_filename.replace('.fasttext', '.fth')
        print("Load clusterized data from", predictions_filename)
        clusterized = load_model_predictions(predictions_filename)

        print('Find named entities in answers...')
        qp = DummyQuestionParser()
        matched = load_matched_data(MATCHED_DATA_FILE)
        matched = clusters4matched(matched, clusterized)

        matched['question_entities_dummy'] = matched.progress_apply(
            lambda row: qp.get_entities(row.question.lower(), row.subject_decoded.lower(), row.object_decoded.lower()),
            axis=1)
        matched = matched[matched.question_entities_dummy != '']
        print("There is answered questions with known named entities and clusters numbers of shape:", matched.shape)

        print('Load the classifier...')
        clf = FastTextClassifier(path=classifier)
        predictions, proba = clf.predict(matched['question'].values)
        matched['pred'] = predictions
        matched['proba'] = proba

        def find_answer(cluster_number, named_entities, strict=True, dummy=True):

            if dummy:
                named_entity = named_entities.lower()
                cluster = clusterized[clusterized.cluster == cluster_number]
                exact_answer = cluster[cluster._subject == named_entity]
                exact_answer = pd.concat([exact_answer, cluster[cluster._object == named_entity]])
                if exact_answer.empty:
                    return ''
                return [(exact_answer.iloc[i]._subject, exact_answer.iloc[i]._relation, exact_answer.iloc[i]._object)
                        for i in range(exact_answer.shape[0])]

            named_entities = [ne.lower() for ne in named_entities]
            exact_answer = clusterized[clusterized.apply(lambda row: row.cluster == cluster_number and any(
                [row._object in ne or row._subject in ne for ne in named_entities]), axis=1)]
            if not exact_answer.empty:
                return exact_answer
            if not strict:
                return clusterized[clusterized.apply(lambda row: row.cluster == cluster_number and any(
                    [compare(ne, row._object) or compare(ne, row._subject) for ne in named_entities]), axis=1)]
            return pd.DataFrame([])

        print("Answer the questions...")
        matched['predicted_answer'] = matched.progress_apply(lambda row: find_answer(row.pred,
                                                                                     row.question_entities_dummy,
                                                                                     dummy=True), axis=1)

        print(f"Found some answer for {matched[matched.predicted_answer != ''].shape[0]} questions of {matched.shape[0]}.")
        matched['eval'] = matched.apply(lambda row: evaluate(row._subject, row._object, row.predicted_answer),
                                        axis=1)

        print("Accuracy for cluster prediction:", matched[matched.pred == matched.cluster].shape[0] / matched.shape[0])
        true_cluster_prediction = matched[matched.pred == matched.cluster]
        true_ans = true_cluster_prediction[true_cluster_prediction['eval'] == True]
        print(f"Accuracy for matched examples with true predicted cluster number (must be 100%): {true_ans.shape[0] / true_cluster_prediction.shape[0]}")

        print("Overall question answering accuracy:", true_ans.shape[0] / matched.shape[0])
        with open(output_file, 'a') as f:
            f.write(f"{predictions_filename}\n")
            f.write(
                f"Accuracy for cluster prediction: {matched[matched.pred == matched.cluster].shape[0] / matched.shape[0]}\n")
            f.write(
                f"Accuracy for matched examples with true predicted cluster number (must be 100%): {true_ans.shape[0] / true_cluster_prediction.shape[0]}\n")
            f.write(f"Overall question answering accuracy: {true_ans.shape[0] / matched.shape[0]}\n\n")


if __name__ == "__main__":
    fire.Fire(test_qa)
