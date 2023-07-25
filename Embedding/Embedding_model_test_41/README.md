# Embeddings for test set
Generating embeddings for textual data is a key process when scientists dig more features, research more works, and explore further development of a project about natural language processing(NLP).
In this term, we will use fine-tuning models (based on Transformer architecture) to generate embeddings.

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
Now you know we have repositories names and labels in **REPO_test.pkl** in **Dataset** folder.
The next thing is extracting series of information (codes, docs, requirements, readme, and structure) using _inspect4py_ on test set.

**Extract_information_test_315.ipynb**: The implementation of extracting information on test set.

After running this Jupyter Notebook, you can get **repo_info_test.pkl** containing some essential information of repositories in test set.
Due to GitHub's limitations on individual file sizes, we will put a Google Drive link about **repo_info_test.pkl** into **Appendix.md** in **Embedding** folder.

Such essential information will be shown on the following:
* **codes**
* **docs**
* **structure**
* **requirements**
* **readme**
* **topic**

### 2. Generating different-level embeddings using the best models.
After we get essential information of each repository, we will use the best fine-tuning models to generate different-level embeddings for validating the validity of evaluation result from train set and validation set.

In each sub-folder named **"ooo_embedding_evaluation_test_315"**, where **"ooo"** means a level, you can find the following four things:
* **UniXcoder**: A pre-trained model can be adapted to such fine-tuning model based on UniXcoder.
* **Evaluation Jupyter Notebook**: A notebook for evaluating on different-level embeddings.
* **Evaluation result**: A ".png" file for describing the ROC curve on different-level embeddings.
* **Similarity calculation**: A ".csv" file for saving similarity calculation information on different-level embeddings by using the best fine-tuning models.

We can get embedding information on different-level. Those information can be found on the following file (Google link will be **Appendix.md**):
* **repo_info_test_code_embeddings.pkl**
* **repo_info_test_doc_embeddings.pkl**
* **repo_info_test_structure_embeddings.pkl**
* **repo_info_test_requirements_embeddings.pkl**
* **repo_info_test_readme_embeddings.pkl**

### 3. Embeddings aggregation
In the second step, we get 5 files described different-level embeddings using the best models.
In this step, we will aggregate those 5 files into 1 aggregation file.

**Embedding_aggregation_test_315.ipynb**: The implementation of aggregating 5 files into 1 file.

**repo_info_test_embeddings.pkl**: The result of aggregation, the link will be putted into **Appendix.md**.

### 4. Evaluation
The second step shows the evaluation results about the best models. 
In this step, we will evaluate the validity of those best models and calculate similarity in each level.

**Embedding_similarity_with_best_models_test.ipynb**: The implementation of calculating similarity by using those best models.

**Embedding_similarity_with_best_model_test.csv**: The similarity calculation result on different-level embeddings by using those best models.

## Importance
Each ".pkl" file is large, especially the one on code embeddings, and it is recommended to use the **reduced** version directly (see **Appendix.md**)