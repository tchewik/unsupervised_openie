cd src/clustering
python train_models.py --idec true
cd ../tests
python test_clustering.py > clustering_tests.txt
python train_classifiers.py --question2cluster_strategy max
python test_qa.py --strategy max