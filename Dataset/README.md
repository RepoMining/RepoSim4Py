# Content
## Importance: Note each pickle (".pkl") file is a python "dict" !!!
## Original data
```bash
├── README.md
├── REPOS.pkl # (repository, label) pairs
├── REPOS_test.pkl # (repository, label) pairs -> test
├── REPOS_train.pkl # (repository, label) pairs -> train
├── REPOS_validation.pkl # (repository, label) pairs -> validation
├── all_data.json # The original json data
└── dataset_processing.ipynb # Processing dataset
```
## Generated data

### Repository information
* By using _**inspect4py**_ on each repository, we can generate such features for each repository.
  * codes
  * docs
  * structure
  * requirements
  * readme
  * topic
* We finally generated **three** **pickle** **files** of **train, validation, test set**.
  * **repo_info_train.pkl** --> link
  * **repo_info_validation.pkl** --> link
  * **repo_info_test.pkl** --> link
* You can read each _**pickle**_ file, and using the following way to access each feature of repository. For example, using **"xxx"** to represent a data set:
  * **repo_info_xxx["repo_name"]["codes"]**: code list
  * **repo_info_xxx["repo_name"]["docs"]**: doc list
  * **repo_info_xxx["repo_name"]["structure"]**: structure list
  * **repo_info_xxx["repo_name"]["requirements"]**: requirement list
  * **repo_info_xxx["repo_name"]["readme"]**: readme list
  * **repo_info_xxx["repo_name"]["topic"]**: repository topic

### Repository embedding information
* Based on **Repository information**, we also generated a series of **pickle** files for describing **embedding** information.
  * train set
    * **repo_info_train_code_embeddings.pkl** --> link
    * **repo_info_train_doc_embeddings.pkl** --> link
    * **repo_info_train_structure_embeddings.pkl** --> link
    * **repo_info_train_requirements_embeddings.pkl** --> link
    * **repo_info_train_readme_embeddings.pkl** --> link
    * **repo_info_train_embeddings.pkl** --> link
  * validation set
    * **repo_info_validation_code_embeddings.pkl** --> link
    * **repo_info_validation_doc_embeddings.pkl** --> link
    * **repo_info_validation_structure_embeddings.pkl** --> link
    * **repo_info_validation_requirements_embeddings.pkl** --> link
    * **repo_info_validation_readme_embeddings.pkl** --> link
    * **repo_info_validation_embeddings.pkl** --> link
  * test set
    * **repo_info_test_code_embeddings.pkl** --> link
    * **repo_info_test_doc_embeddings.pkl** --> link
    * **repo_info_test_structure_embeddings.pkl** --> link
    * **repo_info_test_requirements_embeddings.pkl** --> link
    * **repo_info_test_readme_embeddings.pkl** --> link
    * **repo_info_test_embeddings.pkl** --> link
* The above files were generated because of a lack of computing power, so embedding had to be done separately for each dataset. Using **"xxx"** to represent the type of dataset, and using **"ooo"** to represent the type of embedding.
You can access information what you wanted by the following way:
  * **repo_info_xxx_ooo_embeddings["repo_name"]["codes"]**: using the same way to access essential feature
  * **repo_info_xxx_ooo_embeddings["repo_name"]["ooo_embeddings"]**: **repo_info_xxx_ooo_embeddings.pkl** file can only access **ooo_embeddings**, because of the lack of computing power.
  * **repo_info_xxx_embeddings["repo_name"]["ooo_embeddings"]**: each **repo_info_xxx_embeddings.pkl** is an aggregation of all embeddings. You can assess all embeddings using these **pickle** files.