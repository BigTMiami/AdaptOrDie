# IMBD
curl -Lo train.jsonl https://allennlp.s3-us-west-2.amazonaws.com/dont_stop_pretraining/data/imdb/train.jsonl --output-dir 'datasets/classifier/imdb'
curl -Lo dev.jsonl https://allennlp.s3-us-west-2.amazonaws.com/dont_stop_pretraining/data/imdb/dev.jsonl --output-dir 'datasets/classifier/imdb'
curl -Lo test.jsonl https://allennlp.s3-us-west-2.amazonaws.com/dont_stop_pretraining/data/imdb/test.jsonl --output-dir 'datasets/classifier/imdb'

# AMAZON
curl -Lo train.jsonl https://allennlp.s3-us-west-2.amazonaws.com/dont_stop_pretraining/data/amazon/train.jsonl --output-dir 'datasets/classifier/amazon'
curl -Lo dev.jsonl https://allennlp.s3-us-west-2.amazonaws.com/dont_stop_pretraining/data/amazon/dev.jsonl --output-dir 'datasets/classifier/amazon'
curl -Lo test.jsonl https://allennlp.s3-us-west-2.amazonaws.com/dont_stop_pretraining/data/amazon/test.jsonl --output-dir 'datasets/classifier/amazon'