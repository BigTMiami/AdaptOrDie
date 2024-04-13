# Adapt or Die Project

# Datasets
* Amazon Review Dataset 25M (condenced): BigTMiami/amazon_25M_reviews_condensed 
* Amazon Review Dataset 5M (condenced): BigTMiami/amazon_split_25M_reviews_20_percent_condensed
* Helfullness Dataset: BigTMiami/amazon_helpfulness
* IMDB Dataset: BigTMiami/imdb_sentiment_dataset

# Reproducing Gururangan et. al on Review Dataset
* Notebooks
  * IMDB_Classification
  * Helfullness_Classification

| Experiment                              | Classification Task | Dataset                                                 | Model                                                                          |
| --------------------------------------- | ------------------- | ------------------------------------------------------- | ------------------------------------------------------------------------------ |
| Amazon Review Domain Pre-Training (25M) | \---                | BigTMiami/amazon_25M_reviews_condensed                  | ltuzova/amazon_domain_pretrained_model                                         |
| Amazon Review Domain Pre-Training (5M)  | \---                | BigTMiami/amazon_split_25M_reviews_20_percent_condensed | BigTMiami/amazon_pretraining_5M_model_corrected                                |
| Basline Classifier (RoBERTa)            | IMDB Sentiment      | BigTMiami/imdb_sentiment_dataset                        | ltuzova/imdb_classification_roberta_best_epoch_f1                              |
| Basline Classifier (RoBERTa)            | Amazon Helpfulness  | BigTMiami/amazon_helpfulness                            | ltuzova/amazon_helpfulness_classification_roberta_best_f1                      |
| Classifier (Domain Pre-Pretraining 5M)  | IMDB Sentiment      | BigTMiami/imdb_sentiment_dataset                        | ltuzova/imdb_classification_on_5M_full_pretrained_best_epoch_f1                |
| Classifier (Domain Pre-Pretraining 5M)  | Amazon Helpfulness  | BigTMiami/amazon_helpfulness                            | ltuzova/amazon_helpfulness_classification_on_5M_full_pretrained_best_epoch_f1  |
| Classifier (Domain Pre-Pretraining 25M) | IMDB Sentiment      | BigTMiami/imdb_sentiment_dataset                        | ltuzova/imdb_classification_on_25M_full_pretrained_best_epoch_f1               |
| Classifier (Domain Pre-Pretraining 25M) | Amazon Helpfulness  | BigTMiami/amazon_helpfulness                            | ltuzova/amazon_helpfulness_classification_on_25M_full_pretrained_best_epoch_f1 |

### Metrics

F1-macro for downstream classification (testing set, best epoch):

|             | Basline Classifier (RoBERTa) | DAPT (5M) | DAPT (25M) |
| ----------- | ---------------------------- | --------- | ---------- |
| Helpfulness | 68.91                        | 70.00     | 69.89      |
| IMDB        | 95.16                        | 95.33     | 95.70      |

Comparison with results reported in the Table 5 of the paper (improvement in percent points and % against RoBERTa baseline):

|                                    | Our Results     | Paper           |
| ---------------------------------- | --------------- | --------------- |
| Helfullness: DAPT (5M) vs RoBERTa  | 1.1 pp (0.016%) | \---            |
| Helfullness: DAPT (25M) vs RoBERTa | 1 pp (0.014%)   | 1.4 pp (0.022%) |
| IMDB: DAPT (5M) vs RoBERTa         | 0.2 pp (0.002%) | \---            |
| IMDB: DAPT (25M) vs RoBERTa        | 0.5 pp (0.006%) | 0.4 pp (0.004%) |

Masked LM loss on held-out Review documents before and after domain adaptation: 

|                | DAPT (5M) | DAPT (25M) |
| -------------- | --------- | ---------- |
| RoBERTa        | 1.85      | 1.84       |
| DAPT (Reviews) | 1.52      | 1.41       |

Comparison with results reported in the Table 12 of the paper (decrease in loss):

|                       | Our Results     | Paper           |
| --------------------- | --------------- | --------------- |
| DAPT (5M) vs RoBERTa  | \-0.3 (-0.18%)  | \---            |
| DAPT (25M) vs RoBERTa | \-0.4 (-0.232%) | \-0.2 (-0.081%) |

# Adapters

## Helpfullness Task


| Adapter Type | Pretrain Size | Cls TrainSize | F1 No Pre | F1 With Pre | Notebook                                  | LLM Loss |
| ------------ | ------------- | ------------- | --------- | ----------- | ----------------------------------------- | -------- |
| seq_bn       | 2%            | 10%           | 60.79     | 62.51       | BigTMiami/B_adapter_seq_bn_P_2_C_10.ipynb |          |
| seq_bn       | 3%            | 15%           | 62.37     | 61.36       | BigTMiami/C_adapter_seq_bn_P_3_C_20.ipynb | 2.4084   |
 