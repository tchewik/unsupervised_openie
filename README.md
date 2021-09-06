Run all training & testing scripts (10 models, ~6 hours; ~12 min/model w/o pretrain)

```bash
$ sh train_all.sh
```

Or by steps:

[Step 1] Basic data preparation

```bash
$ cd src/data_preprocessing
src/data_preprocessing$ python simplequestions2wiki.py  # For wikipedia data collection

src/data_preprocessing$  python corenlp_annotation.py --hostname hostname --port 9000 &  # Then, annotate with corenlp
src/data_preprocessing$  python corenlp_annotation.py --hostname hostname --port 9001 --start 10000 &  #  (let's assume we have 30000 text docs and want
src/data_preprocessing$  python corenlp_annotation.py --hostname hostname --port 9002 --start 20000    #  to parse each 10000 at the same time)

src/data_preprocessing$ python basic_features.py --num_threads 16  # Then, get basic features and filter the triplets
```

[Step 2] Prepare data for clustering and evaluation

```bash
src/features$ python assign_embeddings.py --model_name paraphrase-distilroberta-base-v2 --verbose true  # It's time to assign contextual embeddings; ~1.5h
src/features$ python plain_features.py  # Collect final features in output_dir (by default, ../../data/final_features); ~10min
src/features$ python openie2matrices.py  # Make numpy matrices for the clustering
src/data_preprocessing$ python match_openie2simplequestions.py --verbose true  # Then, match basic features with SimpleQuestions dataset
```

[Step 3] Clustering
```bash
src/clustering$ python train_models.py --idec true  # Train models and collect predictions in ../../data/clusterized
src/tests$ python test_clustering.py > clustering_tests.txt
```

[Step 4] Question classification & classification tests
```bash
src/tests$ python train_classifiers.py --question2cluster_strategy {multiclass|max|random}  # ~ 4min/classifier
```
- ``max`` assignes to the WikiData relation in SimpleQuestions the most represented cluster with examples of this relation (from _matched_)
- ``random`` assignes to the relation a random cluster number from the set of clusters we met the examples of this relation
- ``multiclass`` **at the training stage**, solves the task of multiclass classification, assigning to the relation all the clusters we met the examples of this relation

[Step 5] Run question answering test
```bash
src/tests$ python test_qa.py --strategy {multiclass|max|random} --output_file test_results.txt
```
