import numpy as np
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    RobertaForSequenceClassification
)
from sklearn.metrics import accuracy_score,  f1_score

from datasets import load_dataset

from transformers import AutoTokenizer, DataCollatorWithPadding, EarlyStoppingCallback
from transformers import TrainingArguments
from adapters import AdapterTrainer
from transformers import AutoConfig
from adapters import AutoAdapterModel
from adapters import UniPELTConfig, SeqBnConfig

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
    
    
def run_classifier(output_dir, classification_task = 'imdb', pretrained_model = 'roberta-base', \
 torch_compile = True,train = True, evaluate_on_test = True, adapter_hub_name = None, learning_rate=2e-5,
 classification_adapter_name = None, freeze_pretrain = False, resume_from_checkpoint = False,
 weight_decay=0.01):
    print("Early stopping with best val on f1 macro")
    print("Resume from the checkpoint: {0}".format(resume_from_checkpoint))
    print("Classification task: {0}, pretrained model: {1}, lr: {2}, weight_decay: {3}".format(classification_task, 
    pretrained_model, learning_rate, weight_decay))
    
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
    classification_config = AutoConfig.from_pretrained('roberta-base')
    if not adapter_hub_name:
      classification_config.classifier_dropout = 0.1 # From Paper
      classification_config.num_of_labels = 2
      classification_config.id2label=id2label
      classification_config.label2id=label2id
        
    print("Classification config: {0}".format(classification_config))
    
    if not adapter_hub_name:
      model = RobertaForSequenceClassification.from_pretrained(pretrained_model,  config=classification_config)
      if freeze_pretrain:
        for param in model.roberta.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            print(name, param.requires_grad)

    else:
      model = AutoAdapterModel.from_pretrained(pretrained_model,  config=classification_config)

    if adapter_hub_name:
      loaded_adapter_name = model.load_adapter(adapter_hub_name, with_head=False)
      model.add_classification_head(loaded_adapter_name, num_labels=2, id2label=id2label)

      if not classification_adapter_name:
        # Activate the adapter
        model.set_active_adapters(loaded_adapter_name) 

        # Set the adapter to be used for training
        model.train_adapter(loaded_adapter_name)
      
      else:
        # Add new adapter for classification
        new_adapter_config = SeqBnConfig() if classification_adapter_name == "seq_bn" else UniPELTConfig()
        new_adapter_name = classification_adapter_name
        new_adapter_hub_name = "classification_{0}_adapter".format(new_adapter_name)

        model.add_adapter(new_adapter_name, config=new_adapter_config)
        model.train_adapter(new_adapter_name)

        # Activate both old and new adapters
        model.set_active_adapters([loaded_adapter_name,new_adapter_name])

      summary = model.adapter_summary()
      print(summary)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate, # Paper: this is for Classification, not domain training
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        # Lyudmila: changed from 3 to 10 -> used for small roberta
        # Lyudmila: changed back to 3 as agreed
        num_train_epochs=10,  
        weight_decay=weight_decay,
        warmup_ratio=0.06, # Paper: warmup proportion of 0.06
        adam_epsilon=1e-6, # Paper 1e-6 (huggingface default 1e-08)
        adam_beta1=0.9, # Paper: Adam weights 0.9
        adam_beta2=0.98, # Paper: Adam weights 0.98 (huggingface default  0.999)
        lr_scheduler_type="linear",
        save_total_limit=2, # Saves latest 2 checkpoints
        push_to_hub=True,
        hub_strategy="checkpoint", # Only pushes at end with save_model()
        # Lyudmila: Changed to true -> ot seems according to repo and paper that they used early stopping and used best model
        # Lyudmila: Changed to false as agreed
        load_best_model_at_end=True, #Set to false - we want the last trained model like the paper
        torch_compile=torch_compile,  # Much Faster
        logging_strategy="steps", # Is default
        logging_steps=100, # Logs training progress
        metric_for_best_model='f1_macro'
    )

    # EarlyStoppingCallback with patience
    early_stopping = EarlyStoppingCallback(early_stopping_patience=3) # from paper
    # callbacks=[early_stopping],
    
    if not adapter_hub_name:
      trainer = Trainer(
          model=model,
          args=training_args,
          train_dataset=dataset["train"],
          eval_dataset=dataset["dev"],
          tokenizer=tokenizer,
          data_collator=data_collator,
          compute_metrics=compute_metrics,
          callbacks=[early_stopping]
      )
    else:
      trainer = AdapterTrainer(
          model=model,
          args=training_args,
          train_dataset=dataset["train"],
          eval_dataset=dataset["dev"],
          tokenizer=tokenizer,
          data_collator=data_collator,
          compute_metrics=compute_metrics,
          callbacks=[early_stopping]
      )      
    
    print("Trainer args: {0}".format(trainer.args))
    
    if train:
      trainer.train(resume_from_checkpoint = resume_from_checkpoint)

    if evaluate_on_test:
      print("Evaluating model on test set")
      test_evaluation_result = trainer.evaluate(dataset["test"])
      print("Test set eval: {0}".format(test_evaluation_result))

    else:
      print("Evaluating model on dev set")
      test_evaluation_result = trainer.evaluate(dataset["dev"])
      print("Dev set eval: {0}".format(test_evaluation_result))
    
    return trainer, model