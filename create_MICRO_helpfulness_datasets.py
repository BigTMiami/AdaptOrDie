from transformers import AutoTokenizer

from datasets import load_dataset


def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True)


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


base_dir = "datasets/classifier/amazon/"
micro_dir = "datasets/classifier/amazon_micro/"
filenames = ["train.jsonl", "test.jsonl", "dev.jsonl"]

for filename in filenames:
    with open(base_dir + filename, "r") as f:
        data = f.readlines()

    with open(micro_dir + filename, "w") as f:
        f.writelines(data[:5000])


dataset = load_dataset(
    "json",
    data_files={
        "train": "datasets/classifier/amazon_micro/train.jsonl",
        "test": "datasets/classifier/amazon_micro/test.jsonl",
        "dev": "datasets/classifier/amazon_micro/dev.jsonl",
    },
)

dataset
dataset["train"][0]

df = dataset["train"].to_pandas()
labels = df["label"].unique().tolist()
labels

label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for i, label in enumerate(labels)}
print(label2id)
print(id2label)

# Update labels
dataset = dataset.map(lambda examples: {"label": label2id[examples["label"]]})

# Tokenize the dataset
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
block_size = tokenizer.model_max_length

dataset_tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["id", "text"])

dataset_tokenized
dataset_tokenized["train"][0]
dataset["train"][0]["text"]
tokenizer.decode(dataset_tokenized["train"][0]["input_ids"])

datset_name = "amazon_MICRO_helpfulness_dataset"
dataset_tokenized.push_to_hub(datset_name)


# Need to run this to get rid of label column
dataset_tokenized_2 = dataset.map(tokenize_function, batched=True, remove_columns=["id", "text", "label"])

# Condense the dataset
condensed_dataset = dataset_tokenized_2.map(
    group_texts,
    batched=True,
    batch_size=346,
    num_proc=4,
)

condensed_dataset
condensed_dataset["train"][0]

dataset["train"][0]["text"]
dataset["train"][1]["text"]
dataset["train"][2]["text"]
tokenizer.decode(condensed_dataset["train"][0]["input_ids"])
tokenizer.decode(condensed_dataset["train"][1]["input_ids"])

condensed_dataset.push_to_hub(datset_name + "_condensed")
