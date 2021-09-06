import glob
import os
import random

import fire
import pandas as pd
from tqdm import tqdm
from unidecode import unidecode

from question_classifiers import FastTextClassifier

SQ_DATA_PATH = '../../data/simplequestions/openie_matched/'
MATCHED_DATA_FILE = '../../data/simplequestions/openie_matched/matched.csv'
CLUSTERS_PREDICTION_DIR = '../../data/clusterized/'
OUTPUT_MODEL_DIR = '../../models/question_classifiers/'

RANDOM_SEED = 42


def load_matched_data(filename):
    matched = pd.read_csv(filename)
    matched['question'] = matched.question.map(lambda row: unidecode(row.strip()))
    matched['_subject'] = matched.subject_decoded.map(lambda row: unidecode(row).lower().strip())
    matched['_object'] = matched.object_decoded.map(lambda row: unidecode(row).lower().strip())
    matched['subj_obj'] = matched._subject + ' ::: ' + matched._object

    print(f"Loaded matched {matched.shape[0]} (subject; object) pairs from SimpleQuestions.")
    return matched


def load_model_predictions(filename):
    clusterized = pd.read_feather(filename)
    clusterized._subject = clusterized._subject.map(lambda row: unidecode(row).lower().strip())
    clusterized._object = clusterized._object.map(lambda row: unidecode(row).lower().strip())
    clusterized['subj_obj'] = clusterized._subject + ' ::: ' + clusterized._object
    clusterized['obj_subj'] = clusterized._object + ' ::: ' + clusterized._subject

    print(f"Loaded cluster predictions for {clusterized.shape[0]} (subject; object) pairs.")
    return clusterized


def clusters4matched(matched, clusterized):
    _matched = matched.copy()
    _clusterized = clusterized.copy()
    m1 = pd.merge(_matched, _clusterized[['subj_obj', 'cluster', '_relation']], on='subj_obj', how='left')
    m2 = pd.merge(_matched, _clusterized[['obj_subj', 'cluster', '_relation']], left_on='subj_obj', right_on='obj_subj',
                  how='left')
    _matched['cluster'] = m1.cluster.fillna(m2.cluster)
    _matched['relation'] = m1._relation.fillna(m2._relation)
    return _matched


def overlap(true_list, variant_lists):
    result = [value for value in true_list if value in variant_lists]
    if result:
        return result[0]
    return true_list[0]


def train_classifiers(question2cluster_strategy='max',
                      quantize=False):
    for filename in glob.glob(os.path.join(CLUSTERS_PREDICTION_DIR, '*.fth')):
        print()
        print('Looking at the', filename)

        data = {}
        for part in ["train", "valid", "test"]:
            path = os.path.join(SQ_DATA_PATH, f"{part}.csv")
            data[part] = pd.read_csv(path)
            data[part]['assigned_cluster'] = -1
            data[part]['question'] = data[part].question.map(lambda row: unidecode(row))

        clusterized = load_model_predictions(filename)
        matched = load_matched_data(MATCHED_DATA_FILE)
        matched = clusters4matched(matched, clusterized)
        print("Matched pairs are of shape", matched[matched.cluster.notna()].shape)

        print("Assign clusters to the SimpleQuestion dataset...")
        for unique_property in tqdm(matched.property_decoded.unique()):
            df = matched[matched.property_decoded == unique_property]

            # # In case there is no such property in training data
            # # (Uncomment in case of the fault)
            # if not data['train'][data['train'].property_decoded == unique_property].shape:
            #     data['train'] = data['train'].append(df.sample(1))

            for part in data.keys():
                if question2cluster_strategy == 'max':
                    data[part].loc[data[part].property_decoded == unique_property, 'assigned_cluster'] = str(
                        df.cluster.value_counts().keys()[0])

                elif question2cluster_strategy == 'random':
                    data[part].loc[data[part].property_decoded == unique_property, 'assigned_cluster'] = random.choice(
                        df.cluster.unique(),
                    )

                elif question2cluster_strategy == 'multiclass':
                    data[part].loc[data[part].property_decoded == unique_property, 'assigned_cluster'] = ' '.join(
                        map(str, df.cluster.unique().tolist()))

        #  Cheating, randomly move cluster examples from matched to training data
        #  cheating because in this case training data will be different for each clustering algorithm. DEPRECATED!
        # for true_cluster in matched.cluster.unique():
        #     df = matched[matched.cluster == true_cluster].sample(1)
        #     df['assigned_cluster'] = df.cluster
        #     df['mapped_rel'] = df.relation
        #     df.drop(columns=['cluster', 'relation'])
        #     data['train'] = data['train'].append(df)

        print("Training the question classifier...")

        # Here, we oversample the training data because otherwise fasttext will not find a lot of true labels!
        oversample_size = data['train'].assigned_cluster.value_counts().values[1]
        for cluster in data['train'].assigned_cluster.unique():
            df = data['train'][data['train'].assigned_cluster == cluster]
            if df.shape[0] < oversample_size:
                df = df.sample(oversample_size, replace=True, random_state=RANDOM_SEED)
                data['train'] = data['train'].append(df)

        if question2cluster_strategy == 'multiclass':
            unique_clusters = []
            for train_clusters_value in data['train'].assigned_cluster.unique():
                unique_clusters += train_clusters_value.split()
            print("Unique labels in train:", len(list(set(unique_clusters))))
        else:
            print("Unique labels in train:", len(data['train'].assigned_cluster.unique()))

        clf = FastTextClassifier(multilabel=(question2cluster_strategy == 'multiclass'))
        clf.train(data['train']['question'].values, data['train']['assigned_cluster'].values,
                  matched['question'].values, matched.cluster.values)  # cheating again, DEPRECATED
                  # data['valid']['question'].values, data['valid']['assigned_cluster'].values)

        if quantize:
            print("Quantize the model...")
            clf.model.quantize(retrain=False)

        if not os.path.isdir(OUTPUT_MODEL_DIR):
            os.mkdir(OUTPUT_MODEL_DIR)

        output_dir = os.path.join(OUTPUT_MODEL_DIR, question2cluster_strategy)
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        fasttext_model_name = os.path.join(output_dir, filename.split('/')[-1].replace('.fth', '.fasttext'))
        clf.save(fasttext_model_name)
        print("Saved the model in", fasttext_model_name)
        print()

        for part in ['valid', 'test']:
            if question2cluster_strategy == 'multiclass':
                predictions, proba = clf.predict(data[part]['question'].values, multiple_predictions=True, k=1)
                data[part]['_assigned'] = data[part].assigned_cluster.map(lambda row: list(map(int, row.split())))
                data[part]['predictions'] = predictions
                predictions_mono = data[part].apply(lambda row: overlap(row.predictions, row._assigned), axis=1)
                assigned_value = data[part].apply(lambda row: overlap(row._assigned, row.predictions), axis=1)
                report = clf.evaluate(assigned_value, predictions_mono)
            else:
                predictions, proba = clf.predict(data[part]['question'].values, multiple_predictions=False)
                report = clf.evaluate(data[part]['assigned_cluster'].astype(int).values, predictions)

            report_df = pd.DataFrame(report).transpose()
            report_df.to_csv(fasttext_model_name.replace('.fasttext', f'_eval.{part}.csv'))

        part = 'matched'
        if question2cluster_strategy == 'multiclass':
            predictions, proba = clf.predict(matched['question'].values, multiple_predictions=True, k=1)
            matched['predictions'] = predictions
            predictions_mono = matched.apply(lambda row: overlap(row.predictions, [row.cluster]), axis=1)
            assigned_value = matched.apply(lambda row: overlap([row.cluster], row.predictions), axis=1)

            report = clf.evaluate(assigned_value, predictions_mono)
        else:
            predictions, proba = clf.predict(matched['question'].values, multiple_predictions=False)
            report = clf.evaluate(matched['cluster'].astype(int).values, predictions)

        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(fasttext_model_name.replace('.fasttext', f'_eval.{part}.csv'))
        print("Evaluation on matched (real clusters):")
        print(report_df)


if __name__ == '__main__':
    fire.Fire(train_classifiers)
