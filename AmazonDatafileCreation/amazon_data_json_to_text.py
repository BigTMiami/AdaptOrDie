import json
import os

from amazon_create_download_scripts import create_dir
from amazon_dataset_category_name import amazon_datasets


def load_reviews(json_file):
    # This function will skip reviews without review text and video reviews
    # It will replace newlines with spaces, remove double newlines and remove double quotes
    # If this review is the same as the previous review, it will be skipped
    reviews = []
    previous_review = ""

    with open(json_file, "r") as fp:
        line_no = 0
        for line in fp:
            line_no += 1
            entry = json.loads(line.strip())
            if "reviewText" not in entry:
                continue
            review = entry["reviewText"].strip()
            if "video-block" in review:
                continue
            review = review.replace("\n\n", " ").strip()
            review = review.replace('"', " ").strip()
            review = review.replace("\n", " ").strip()

            if review == previous_review or len(review) == 0 or review is None:
                continue
            previous_review = review
            reviews.append(review + "\n")
    return reviews


def write_reviews(reviews_file, reviews):
    with open(reviews_file, "w") as fp:
        fp.writelines(reviews)


def convert_all_json_to_text(
    json_dir="datasets/domain/amazon_json/",
    txt_dir="datasets/domain/amazon_reviews/",
    json_suffix=".json",
    txt_dir_suffix=".txt",
):
    create_dir(txt_dir)

    for name in amazon_datasets:
        json_file = f"{json_dir}{name}{json_suffix}"
        reviews_file = f"{txt_dir}{name}{txt_dir_suffix}"

        if os.path.isfile(json_file):
            print(f"Writing {reviews_file} from  {json_file}")

            reviews = load_reviews(json_file)
            write_reviews(reviews_file, reviews)
        else:
            print(f"ERROR: File not found: {json_file}")


if __name__ == "__main__":
    convert_all_json_to_text()
