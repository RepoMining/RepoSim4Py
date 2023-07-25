# Embeddings for train set

## Pre-conditions
In order to generate embedding for repositories, observing dataset is an essential thing.
Before you read the following content, you need to read the **README.md** in **Dataset** folder.

## Requirements
* inspect4py
* torch
* pandas
* transformers
* sentence-transformers
* tqdm
* scikit-learn
* matplotlib

## Implementation Process
What follows is written in the order of the implementation process.
### 1. Extracting information
Now you know we have repositories names and labels in **REPO_train.pkl** in **Dataset** folder.
The next thing is extracting series of information (codes, docs, requirements, readme, and structure) using _inspect4py_ on train set.

**Extract_information_train_315.ipynb**: The implementation of extracting information on train set.

After running this Jupyter Notebook, you can get **repo_info_train.pkl** containing some essential information of repositories in train set.
Due to GitHub's limitations on individual file sizes, we will put a Google Drive link about **repo_info_train.pkl** into **Appendix.md** in **Embedding** folder.

Such essential information will be shown on the following:
* **codes**
* **docs**
* **structure**
* **requirements**
* **readme**
* **topic**

### 2. Generating different-level embeddings using different models.
After we get essential information of each repository, we can use different fine-tuning models to generate different-level embeddings.
In each sub-folder named **"ooo_embedding_evaluation_train_315"**, where **"ooo"** means a level, you can find the following four things:
* **UniXcoder**: A pre-trained model can be used to such fine-tuning model based on UniXcoder.
* **Evaluation Jupyter Notebook**: A notebook for evaluating on different-level embeddings.
* **Evaluation result**: A ".png" file for describing the ROC curve on different-level embeddings.
* **Similarity calculation**: A ".csv" file for saving similarity calculation information on different-level embeddings by using different fine-tuning models.

We can get embedding information on different-level by using different models. Those information can be found on the following file (Google link will be **Appendix.md**):
* **repo_info_train_code_embeddings.pkl**
* **repo_info_train_doc_embeddings.pkl**
* **repo_info_train_structure_embeddings.pkl**
* **repo_info_train_requirements_embeddings.pkl**
* **repo_info_train_readme_embeddings.pkl**

### 3. Embeddings aggregation
In the second step, we get 5 files described different-level embeddings using different models.
In this step, we will aggregate those 5 files into 1 aggregation file.

**Embedding_aggregation_train_315.ipynb**: The implementation of aggregating 5 files into 1 file.

**repo_info_train_embeddings.pkl**: The result of aggregation, the link will be putted into **Appendix.md**.

### 4. Choosing the best model
The second step shows the evaluation results about different models. 
In this step, we will choose the best model for calculation similarity.

**Embedding_similarity_with_best_models_train.ipynb**: The implementation of choosing best models and calculating similarity.

**Embedding_similarity_with_best_model_train.csv**: The similarity calculation result on different-level embeddings by using the best model.

## Importance
Each ".pkl" file is large, especially the one on code embeddings, and it is recommended to use the **reduced** version directly (see **Appendix.md**)