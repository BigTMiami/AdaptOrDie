import gzip
import os
import shutil

from amazon_create_download_scripts import create_dir
from amazon_dataset_category_name import amazon_datasets


def unzip_data_files(
    data_dir="datasets/domain/amazon_raw/",
    json_dir="datasets/domain/amazon_json/",
    raw_suffix="_5.json.gz",
    json_suffix=".json",
):

    create_dir(json_dir)

    for name in amazon_datasets:
        gz_file = f"{data_dir}{name}{raw_suffix}"
        json_file = f"{json_dir}{name}{json_suffix}"

        if os.path.isfile(gz_file):
            print(f"Unzipping{json_file} ...")
            with gzip.open(gz_file, "rb") as f_in:
                with open(json_file, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
        else:
            print(f"ERROR: File not found: {gz_file}")


if __name__ == "__main__":
    unzip_data_files()
