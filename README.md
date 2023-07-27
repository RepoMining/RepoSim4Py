# RepoSim4Py

## Description

RepoSim4Py is a project for determining the semantic similarity among python repositories based on embedding approach.
This project contains series of `Jupyter Notebook` and `Python Script` conducted for our approach to detect semantically
similar python repositories using different models based on embedding approach.

These `Jupyter Notebook` files include the whole process for choosing (evaluating) the best model considering `Doc2Vec`
models and `pre-trained` models based on `Transformers` architecture.
However, the rest `Python Script` files are the process of generating different-level embeddings and calculating cosine
similarities by using the best models.

We considered different-level information of a repository: `codes`, `docstrings`, `requirements`, `readmes`,
and `structures`.
Currently, on each level (apart from `structure` level, we discard this level at the end), our best performing model
is [UniXCoder](https://github.com/microsoft/CodeBERT/tree/master/UniXcoder/downstream-tasks/code-search#1-advtest-dataset)
fine-tuned on code search task with `AdvTest` dataset.

More details about model evaluations can be found in `Doc2Vec` and `Embedding` folder.
Moreover, more details on our approach's implementations and applications can be found in the `Script` folder.

## Features

Given a list of python repositories' name, RepoSim4Py can:

* Extracting all `codes`, `docstrings`, `requirements`, and `readmes` from the given repositories.
* Generating the embeddings for each level (as mentioned as before) using a `UniXCoder` model fine-tuned on the
  nl-code-search task.
* Aggregating these 4 level `mean` embeddings into 1 repo-level `mean` embedding.
* Calculating semantic similarities between 2 (at least 2) repositories on different level
  embeddings (`codes-level`, `docs-level`, `requirements-level`, `readmes-level`, and `repo-level`).
* Saving repositories information (such as codes, docs, code embeddings, repository embeddings, etc.) into a `.pkl`
  file, and semantic similarity calculation results into a `.csv` file.

## Installation

### Prerequisites

* Python 3.9+
* pip

### Package dependencies

```
accelerate==0.21.0
torch==2.0.1
numpy==1.25.0
pandas==2.0.3
transformers==4.30.2
sentence-transformers==2.2.2
tqdm==4.65.0
inspect4py==0.0.8
scikit-learn==1.3.0
matplotlib==3.7.2
```

### Installation from code

1. **Clone the project**: You can download the source code using the following instruction.

```bash
git clone https://github.com/PythonSimilarity/RepoSim4Py.git
```

2. **Install package dependencies**: Install the required dependencies by running the following command.

```bash
pip install -r requirements.txt
```

When you follow the instructions above, you will have reached the basic conditions for running this project.

## Usage

You need to change your current directory to `Script` folder before you enjoy the above features.

```bash
cd Script
```

and then using the following format to run this program.

```
python RepoSim4Py.py --input <repo1> <repo2> ... --output <output_dir> [--eval]
```

For example:

```bash
python RepoSim4Py.py -i lepture/authlib idan/oauthlib evonove/django-oauth-toolkit selwin/python-user-agents SmileyChris/django-countries django-compressor/django-compressor billpmurphy/hask pytoolz/toolz Suor/funcy przemyslawjanpietrzak/pyMonet -o output/ --eval
```

The input is a list of GitHub repository names (at least 2) in the format of `<owner>/<repo>` (e.g. `lepture/authlib`). 
The output of the script is a python pickle file `<output_dir>/output.pkl` which stores a list of dictionaries containing all repositories' information, 
including name, topics, license, stars, extracted `codes/docstrings/requirements/readmes` list, embeddings corresponding to extracted information, mean embedding corresponding to extracted information, as well as the repo-level mean embedding. 
This file can be used for later experiments such as semantic similarity calculation/comparison.

When `--eval` is specified, the script will also save a csv file with 9 columns: `repo1`, `repo2`, `topics1`, `topics2`, `code_sim`, `doc_sim`, `requirement_sim`, `readme_sim`, and `repo_sim`, 
representing two repositories and their similarity scores in terms of different-level mean embeddings. This file will compare each pair of repositories in the input list and save the results at `<output_dir>/evaluation_result.csv`.


## License

Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgments

* [GraphCodeBERT](https://arxiv.org/abs/2009.08366)
* [UniXCoder](https://arxiv.org/abs/2203.03850)
* [Sentence Transformers](https://www.sbert.net/)
* [Doc2Vec](https://arxiv.org/abs/1405.4053)
* [AdvTest](https://arxiv.org/abs/1909.09436)
* [awesome-python](https://github.com/vinta/awesome-python/)
* [Original work of the customized GraphCodeBERT model by @snoop2head](https://github.com/sangHa0411/CloneDetection)
* [Doc2Vec implementation by @gensim](https://radimrehurek.com/gensim/models/doc2vec.html)