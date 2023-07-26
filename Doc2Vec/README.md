# Doc2Vec for train set
Generating embeddings for textual data is a key process when scientists dig more features, research more works, and explore further development of a project about natural language processing(NLP).
In this term, we will use _Doc2Vec_ model to generate embeddings.

## Pre-conditions
In order to generate embedding for repositories, observing dataset is an essential thing.
Before you read the following content, you need to read the `README.md` in `Dataset` folder.

Besides, this part was originally designed to try to generate different-level embeddings using some models (small models) that are not based on the Transformer architecture.
So it is a necessary thing to be familiar with generating embeddings using models based on the Transformer architecture. 
Please make sure you have read the contents of the `Embedding` folder before viewing everything in this folder.

## Requirements
* gensim
* torch
* scikit-learn
* pandas
* tqdm
* matplotlib

## Description
After exploring the result generated by such fine-tuning models, we want to use _Doc2Vec_ to make a deeper attempted on readme, requirements, and structure level.
Using train set as an example and to be observed. 

The results show that we did not get better results on the training set, so there is no need to continue with the same method on the validation and test sets.

In each sub-folder named `Doc2Vec_ooo_train_315`, where `ooo` means a level, there are four common files you need to consider:
* `Evaluation Jupyter Notebook`: A notebook for evaluating on different-level embeddings.
* `Evaluation result`: A `.png` file for describing the ROC curve on different-level embeddings.
* `Similarity calculation`: A `.csv` file for saving similarity calculation information on different-level embeddings by using the _Doc2Vec_ model.
* `Corpus`: A `.txt` file for describing corpus based each level.