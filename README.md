# WWAggr: A Window Wasserstein-based Aggregation for Ensemble Change Point Detection

Clean code and README are upcoming!

## Dependancies
```pip install -r requirements.txt```

## Datasets
All the data samples are available [online](https://limewire.com/d/MYhOk#qwo4dF6h3k). Once loaded, unzip the `data.zip` file to create the ```/data``` folder.

## Pretrained models
All the pretrained models for ensembles are available [online](https://limewire.com/d/MYhOk#qwo4dF6h3k). Once loaded, unzip the `saved_models.zip` file to create the ```/saved_models``` folder.

## Experiments
* ```train_models.py``` - script for training base models for ensembles from scratch
* ```evaluate_ensembles.py``` - script for standard complete evaluation of a pretrained ensemble with different aggregations
* ```evaluate_thresholds.py``` - script for evaluation of the proposed aggregation procedure with different threshold numbers
