from sklearn.metrics.cluster import v_measure_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from train_classifiers import load_matched_data, load_model_predictions, clusters4matched
import glob
import os
import fire


MATCHED_DATA_FILE = '../../data/simplequestions/openie_matched/matched.csv'
CLUSTERS_PREDICTION_DIR = '../../data/clusterized/'


def test_clustering():
    for filename in glob.glob(os.path.join(CLUSTERS_PREDICTION_DIR, '*.fth')):
        print()
        print('Looking at the', filename)

        clusterized = load_model_predictions(filename)
        if not clusterized.shape[0]:
            print('Empty predictions file.')
        else:
            matched = load_matched_data(MATCHED_DATA_FILE)
            matched = clusters4matched(matched, clusterized)
            print("Matched pairs are of shape", matched[matched.cluster.notna()].shape)

            print("V-measure:", v_measure_score(matched.property_decoded, matched.cluster))
            print("AMI:", adjusted_mutual_info_score(matched.property_decoded, matched.cluster))


if __name__ == '__main__':
    fire.Fire(test_clustering)
