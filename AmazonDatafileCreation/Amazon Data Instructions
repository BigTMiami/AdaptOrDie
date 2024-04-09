# Steps
* Create scripts to check downloads and then do downloads:

        python amazon_create_download_scripts.py

* Update permissions to make scripts runable

        chmod +x amazon_data_check_download_links.sh
        chmod +x amazon_data_download_files.sh

* First run the check to make sure the links work

         ./amazon_data_check_download_links.sh

* Then download files

         ./amazon_data_download_files.sh

* Now there should be the gz files in the datasets/domain/amazon_raw folder

* Unzip these with the script

         python amazon_data_unzip_downloads.py

* You may now want to delete the zipped file directory to save space

        rm -r datasets/domain/amazon_raw

* Now pull out just the review text with a little cleaning

        python amazon_data_json_to_text.py

* Once again you can remove the unneeded json files

        rm -r datasets/domain/amazon_json

* Now you have a file with all the text reviews on one line each in category files in amazon_reviews

* To create master domain training files, use the create_amazon_domain_dataset_function_script.py.  It has 5 arguements
    * **output_file_basename**: this is the base name for the dataset.  It will have appended to it the size of the training set.  This will be pushed to the huggingface hub by default. Example:
            output_file_basename='amazon_test'
            train_size = 50000
            dataset name = amazon_test_50_000
    * **train_size**: if int, it will load that many reviews.  if float between 0 and 1, it reads that as a percentage
    * **validation_size**=None: if an int is used, it creates a validation set of that many reviews in the dataset
    * **test_size**=None: if an int is used, it creates a test set of that many reviews in the dataset
    * **push_to_hub**=True: will push to hub.  You need to have logged into huggingface using huggingface-cli login on your machine to do this
    * **local_save_filename**=None:  if this is set, it will save a local dataset file to this location

* This example that creates a dataset with a train set of 10,000 reviews and a validation set of 5,000.  This will load a dataset called **amazon_test_set_10_000** to the huggingface hub


                python create_amazon_domain_dataset_function_script.py amazon_test_set 10000 -val 5000