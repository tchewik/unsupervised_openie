import glob
import os
import numpy as np
from train_classifiers import load_matched_data
from sklearn.metrics.cluster import v_measure_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
import fire

MATCHED_DATA_FILE = '../../data/simplequestions/openie_matched/matched.csv'
CLUSTERS_PREDICTION_DIR = '../../data/clusterized/'


def random_unif_pred(n, length):
    return np.random.randint(0, n + 1, size=length)


def random_exp_pred(n, length):
    predicted = np.random.exponential(scale=0.4, size=length)
    predicted = predicted / predicted.max() * n
    predicted = predicted.astype(int)

    return predicted


def test_clustering():
    matched = load_matched_data(MATCHED_DATA_FILE)
    for filename in glob.glob(os.path.join(CLUSTERS_PREDICTION_DIR, '*.fth')):
        num_clusters = int(filename.split('_clusters_')[1].split('_')[0])
        matched['cluster_uniform'] = random_unif_pred(num_clusters, matched.shape[0])
        matched['cluster_exp'] = random_exp_pred(num_clusters, matched.shape[0])

        print(filename)
        print("(uniform) V-measure:", v_measure_score(matched.property_decoded, matched.cluster_uniform))
        print("(uniform) AMI:", adjusted_mutual_info_score(matched.property_decoded, matched.cluster_uniform))
        print("(exp) V-measure:", v_measure_score(matched.property_decoded, matched.cluster_exp))
        print("(exp) AMI:", adjusted_mutual_info_score(matched.property_decoded, matched.cluster_exp))
        print()


if __name__ == '__main__':
    fire.Fire(test_clustering)