# Dataset: Awesome Python
This dataset contains list of awesome Python frameworks, libraries, software and resources. 
It supports list of names and types of GitHub repositories, which can support researchers and developers to do series of data mining works.
Considering the details of a repository, we can dig collection of textual data such as codes, documents, requirements, README, etc.
These textual data can be used for various natural language processing (NLP) tasks, such as text classification, summarization, and language modeling.

## Dataset Description
* Encoding: UTF-8
* `awesome-python`: https://github.com/vinta/awesome-python

## Original File Description
* `all_data.json`: A JSON file about awesome-python whose key is the category of repository and value is list of repositories that categorize to key.

## Preprocessing Scripts
* `dataset_processing.ipynb`: A Jupyter Notebook to clean invalid repositories based on its name, and to split dataset to train, validation, and test set.

## Final File Description
### Note: each pickle (`.pkl`) file is a python `dict` !!!
* `REPOS.pkl`: Containing all repositories with _(repository_name, label)_ format.
* `REPOS_train.pkl`: Containing train set repositories with _(repository_name, label)_ format.
* `REPOS_validation.pkl`: Containing validation set repositories with _(repository_name, label)_ format.
* `REPOS_test.pkl`: Containing test set repositories with _(repository_name, label)_ format.