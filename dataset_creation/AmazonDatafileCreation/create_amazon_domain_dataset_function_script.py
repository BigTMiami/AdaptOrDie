import argparse
import gc

from transformers import AutoTokenizer

from amazon_create_download_scripts import create_dir
from create_amazon_master_text_files import sample_all_files_multi
from datasets import load_dataset


def create_amazon_dataset(
    output_file_basename,
    train_size,
    validation_size=None,
    test_size=None,
    data_dir="datasets/domain/amazon_reviews/",
    output_dir="datasets/domain/amazon_master/",
    push_to_hub=True,
    local_save_filename=None,
    push_non_condensed=False,  # IF PUSH NON CONDENSED, USE TRUNCATION
):
    create_dir(output_dir)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=truncation)

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    if type(train_size) is float:
        train_size_final = int(25_725_000 * train_size)
    else:
        train_size_final = train_size

    # Truncation is only used if pushing a non-cqndensed dataset
    # If pushing a condensed dataset, truncation is not needed because the grouping function will handle it
    truncation = push_non_condensed

    output_file_basename = f"{output_file_basename}_{train_size_final:_}"

    print(f"Train Size: {train_size_final}")

    file_info = {
        "train": train_size_final,
    }
    if validation_size:
        file_info["validation"] = validation_size
    if test_size:
        file_info["test"] = test_size

    # max_file_testing_cap = 3
    sample_all_files_multi(
        data_dir,
        output_dir,
        output_file_basename,
        file_info,
        # max_file_testing_cap=max_file_testing_cap,
    )

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    block_size = tokenizer.model_max_length

    print("==================================================================")

    datafiles = {}
    for name in file_info.keys():
        datafiles[name] = f"{output_dir}{output_file_basename}_{name}.txt"

    dataset = load_dataset(
        "text",
        data_files=datafiles,
    )
    print("")
    print("Dataset Information")
    print(dataset)

    dataset_tokenized = dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
    print("")
    print("Tokenized Dataset Information")
    print(dataset_tokenized)

    if push_non_condensed:
        print(f"Pushing Tokenized Dataset to hub: {output_file_basename}")
        dataset_tokenized.push_to_hub(output_file_basename)

    del dataset
    gc.collect()

    condensed_dataset = dataset_tokenized.map(
        group_texts,
        batched=True,
        batch_size=346,
        num_proc=4,
    )
    print("Condensed Dataset Information")
    print(condensed_dataset)

    del dataset_tokenized
    gc.collect()

    if push_to_hub:
        output_file_basename = f"{output_file_basename}_condensed"
        print(f"Pushing Condensed Dataset to hub: {output_file_basename}")
        condensed_dataset.push_to_hub(output_file_basename)

    if local_save_filename is not None:
        print(f"Saving Condensed Dataset to {local_save_filename}")
        condensed_dataset.save_to_disk(local_save_filename)

    del condensed_dataset
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Amazon Domain Dataset Creation",
    )
    parser.add_argument("output_file_basename")  # positional argument
    parser.add_argument("train_size")
    parser.add_argument("-validation_size", type=int, default=None)
    parser.add_argument("-test_size", type=int, default=None)
    parser.add_argument("-push_non_condensed", default=False, action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    train_size_string = args.train_size
    if train_size_string.isdigit():
        train_size = int(train_size_string)
    else:
        try:
            train_size = float(train_size_string)
        except:
            print(f"Training Size: {train_size_string} is not a number. Exiting.")
            exit()

    create_amazon_dataset(
        args.output_file_basename,
        train_size,
        validation_size=args.validation_size,
        test_size=args.test_size,
        push_non_condensed=args.push_non_condensed,
    )
