# WWAggr: A Window Wasserstein-based Aggregation for Ensemble Change Point Detection

Clean code and README are upcoming! Currently, the datasets and the pretrained' models weights cannot be published due to the double anonymity policy.

## Dependancies
```pip install -r requirements.txt```

## Datasets
All the data samples are available [TBA](). Once loaded, put them into the ```/data``` folder.

## Pretrained models
All the pretrained models for ensembles are available [TBA](). Once loaded, put them into the ```/saved_models``` folder.

## Experiments
* ```train_models.py``` - script for training base models for ensembles from scratch
* ```evaluate_ensembles.py``` - script for standard complete evaluation of a pretrained ensemble with different aggregations
* ```evaluate_thresholds.py``` - script for evaluation of the proposed aggregation procedure with different threshold numbers
