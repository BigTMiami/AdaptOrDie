# Adapt or Die Project

# Datasets
* Amazon Review Dataset 25M (condenced): BigTMiami/amazon_25M_reviews_condensed 
* Amazon Review Dataset 5M (condenced): BigTMiami/amazon_split_25M_reviews_20_percent_condensed
* Helfullness Dataset: BigTMiami/amazon_helpfulness
* IMDB Dataset: BigTMiami/imdb_sentiment_dataset

# Reproducing Gururangan et. al on Review Dataset

## Domain Pretraining
* Full dataset of ~25M reviews: 
    * Notebook: Amazon_Domain_Full_Dataset_Pre_training_Model.ipynb
    * Model path: ltuzova/amazon_domain_pretrained_model

* Subset of ~5M reviews: 
    * Notebook: Amazon_Domain_Pre_training_5M_Corrected.ipynb
    * Model path: BigTMiami/amazon_pretraining_5M_model_corrected

## Classification Task
### Amazon Helpfullness: 
* Roberta baseline: 
    * Notebook: Amazon_Helpfulness_Classification_Full_Dataset.ipynb
    * Model path: BigTMiami/amazon_helpfulness_classification_on_amazon_5M_model_corrected
* Domain-pretrained on ~5M: 
    * Notebook: Amazon_Helpfulness_Classification_on_5M_pretrained_model_corrected.ipynb
    * Model path: BigTMiami/amazon_helpfulness_classification_full
* Domain-pretrained on ~25M:
    * Notebook: TBD
    * Model path: TBD

### IMDB: 
* Roberta baseline: 
    * Notebook: IMDB_Classification.ipynb
    * Model path: ltuzova/imdb_classification_roberta
* Domain-pretrained on ~5M: 
    * Notebook: IMDB_Classification.ipynb
    * Model path: ltuzova/imdb_classification_on_5M_full_pretrained
* Domain-pretrained on ~25M:
    * Notebook: IMDB_Classification.ipynb
    * Model path: ltuzova/imdb_classification_on_25M_full_pretrained

### Metrics

Comparison on Testing Set (best epoch, F1 Macro) (Table 5 from original paper)

| Domain  | Task         | RoBERTa | DAPT (5M) | DAPT (25M)|
| ------- | ------------ | ------- | --------- | --------- |
| Reviews | Helpfullness | TBD     | TBD       | TBD       |
| Reviews | IMDB         | 95.08   | 95.42     | TBD       |

Metrics comparison on Dev Set (last epoch/best epoch, F1 Macro) (Table 15 from original paper)

| Domain  | Task         | RoBERTa     | DAPT (5M)   | DAPT (25M) |
| ------- | ------------ | ----------- | ----------- | --------- |
| Reviews | Helpfullness | TBD         | TBD         | TBD       |
| Reviews | IMDB         | 94.84/94.64 | 95.08/95.12 | TBD       |

# Adapters