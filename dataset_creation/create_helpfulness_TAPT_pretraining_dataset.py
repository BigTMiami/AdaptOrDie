from transformers import AutoTokenizer

from datasets import load_dataset


def tokenize_function(examples):
    return tokenizer(examples["text"])


def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)] for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


dataset = load_dataset(
    "json",
    data_files={
        "train": "datasets/classifier/amazon/train.jsonl",
        "test": "datasets/classifier/amazon/test.jsonl",
        "dev": "datasets/classifier/amazon/dev.jsonl",
    },
)

dataset

dataset["train"][0]

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
block_size = tokenizer.model_max_length


dataset_tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["id", "text", "label"])


dataset_tokenized
dataset_tokenized["train"]
dataset_tokenized["train"][0]

condensed_dataset = dataset_tokenized.map(
    group_texts,
    batched=True,
    batch_size=346,
    num_proc=4,
)

condensed_dataset
condensed_dataset["train"]
condensed_dataset["train"][0]

condensed_dataset.push_to_hub("amazon_helpfulness_TAPT_pretraining_dataset")
