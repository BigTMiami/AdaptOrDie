# from a_review_datasets import amazon_datasets
import os

from amazon_dataset_category_name import amazon_datasets


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def create_amazon_data_download_scripts(data_dir="datasets/domain/amazon_raw/", filename="amazon_data"):
    # Example output
    # curl -L -k -o train.json.gz  https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/AMAZON_FASHION_5.json.gz!
    create_dir(data_dir)

    website = "https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/"
    suffix = "_5.json.gz"
    check_command = "curl -o /dev/null --silent -Iw '%{http_code}'"
    download_command = "curl -L -k -o "

    check_filename = filename + "_check_download_links.sh"
    with open(check_filename, "w") as f:
        for name in amazon_datasets:
            f.writelines(f"echo {name}\n")
            f.writelines(f"{check_command} {website}{name}{suffix}\n")
            f.writelines(f"echo \n")

    download_filename = filename + "_download_files.sh"
    with open(download_filename, "w") as f:
        for name in amazon_datasets:
            f.writelines(f"echo {name}\n")
            f.writelines(f"{download_command} {data_dir}{name}{suffix} {website}{name}{suffix}\n")
            f.writelines(f"echo \n")


if __name__ == "__main__":
    create_amazon_data_download_scripts()
