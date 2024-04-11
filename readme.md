# Adapt or Die Project

## Datasets
* Amazon Review Dataset 25M (condenced): BigTMiami/amazon_25M_reviews_condensed 
* Amazon Review Dataset 5M (condenced): BigTMiami/amazon_split_25M_reviews_20_percent_condensed
* Helfullness Dataset: BigTMiami/amazon_helpfulness
* IMDB Dataset: BigTMiami/imdb_sentiment_dataset

## Reproducing Gururangan et. al on Review Dataset

### Domain Pretraining
* Full dataset of ~25M reviews: 
    * Notebook: Amazon_Domain_Full_Dataset_Pre_training_Model.ipynb
    * Model path: ltuzova/amazon_domain_pretrained_model

* Subset of ~5M reviews: 
    * Notebook: Amazon_Domain_Pre_training_5M_Corrected.ipynb
    * Model path: BigTMiami/amazon_pretraining_5M_model_corrected

### Classification Task
* Amazon Helpfullness: Roberta baseline: Amazon_Helpfulness_Classification_Full_Dataset.ipynb
* Amazon Helpfullness: Domain-pretrained on ~5M: Amazon_Helpfulness_Classification_on_5M_pretrained_model_corrected.ipynb
