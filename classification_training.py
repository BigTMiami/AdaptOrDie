import numpy as np
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score,  f1_score

from datasets import load_dataset

from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import TrainingArguments
from transformers import AutoConfig

def compute_metrics(pred, average = 'macro'):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    # Calculate accuracy
    accuracy = accuracy_score(labels, preds)

   # Calculate precision, recall, and F1-score
    f1 = f1_score(labels, preds, average=average)

    return {
        'accuracy': accuracy,
        'f1_macro': f1
    }
    
    
def train_classifier(output_dir, classification_task = 'imdb', pretrained_model = 'roberta-base', torch_compile = True):
    print("Classification task: {0}, pretrained model: {1}".format(classification_task, pretrained_model))
    
    if classification_task == 'helpfulness':
        dataset = load_dataset("BigTMiami/amazon_helpfulness")
        
    elif classification_task == 'imdb':
        dataset = load_dataset("BigTMiami/imdb_sentiment_dataset")
        
    print("Dataset: {0}".format(dataset))
        
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    if classification_task == 'helpfulness':
        id2label = {0: "unhelpful", 1: "helpful"}
        label2id = {"unhelpful": 0, "helpful": 1}
        
    elif classification_task == 'imdb':
        # not needed probably, just for simplicity
        id2label = {0: 0, 1: 1}
        label2id = {0: 0, 1: 1}

    # Set Classifier Settings
    classification_config = AutoConfig.from_pretrained(pretrained_model)
    classification_config.classifier_dropout = 0.1 # From Paper
    classification_config.num_of_labels = 2
    classification_config.id2label=id2label
    classification_config.label2id=label2id,
        
    print("Classification config: {0}".format(classification_config))
    
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model,  config=classification_config)

    from transformers import TrainingArguments

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5, # Paper: this is for Classification, not domain training
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10, # NEED TO TRAIN FOR REAL TEST
        weight_decay=0.01,
        warmup_ratio=0.06, # Paper: warmup proportion of 0.06
        adam_epsilon=1e-6, # Paper 1e-6 (huggingface default 1e-08)
        adam_beta1=0.9, # Paper: Adam weights 0.9
        adam_beta2=0.98, # Paper: Adam weights 0.98 (huggingface default  0.999)
        lr_scheduler_type="linear",
        save_total_limit=2, # Saves latest 2 checkpoints
        push_to_hub=True,
        hub_strategy="checkpoint", # Only pushes at end with save_model()
        load_best_model_at_end=False, # Set to false - we want the last trained model like the paper
        torch_compile=torch_compile,  # Much Faster
        logging_strategy="steps", # Is default
        logging_steps=100, # Logs training progress
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["dev"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    print("Trainer args: {0}".format(trainer.args))
    
    trainer.train()
    trainer.evaluate(dataset["test"])
    
    return trainer