"""
Download the embeddings by the link: https://drive.google.com/drive/folders/1HoJ6dJqLU3OZrIx3wkj-LOzrLNpf6iKS?usp=drive
1. repo_info_train_embeddings_reduce.pkl
2. repo_info_validation_embeddings_reduce.pk;
3. repo_info_test_embeddings_reduce.pkl
"""

import pickle
import json
import pandas as pd
import torch
from torch.nn.functional import cosine_similarity
from itertools import combinations
from tqdm.auto import tqdm


def get_mean_embedding(embeddings):
    return torch.mean(embeddings.reshape(-1, 768), dim=0)


def cossim(embedding1, embedding2):
    return cosine_similarity(embedding1, embedding2, dim=0).item()


# 1. Loading embeddings
with open("repo_info_train_embeddings_reduce.pkl", "rb") as f:
    repo_info_train_embeddings = pickle.load(f)
    f.close()

with open("repo_info_validation_embeddings_reduce.pkl", "rb") as f:
    repo_info_validation_embeddings = pickle.load(f)
    f.close()

with open("repo_info_test_embeddings_reduce.pkl", "rb") as f:
    repo_info_test_embeddings = pickle.load(f)
    f.close()

# 2. Observing the data shape
# print(next(iter(repo_info_train_embeddings.values())).keys()) # dict_keys(['topic', 'codes_embeddings', 'docs_embeddings', 'structure_embeddings', 'requirements_embeddings', 'readme_embeddings'])
# print(next(iter(repo_info_train_embeddings.values())).get("codes_embeddings").shape) # embeddings, not mean-embedding

# 3. Merge the embeddings
repo_embeddings_dict = repo_info_train_embeddings
repo_embeddings_dict.update(repo_info_validation_embeddings)
repo_embeddings_dict.update(repo_info_test_embeddings)
# print(len(repo_embeddings_dict)) # 456

# 4. Loading the repo json file
with open("filtered_manual_categories.json", "r") as f:
    categories_repos = json.load(f)

repos_list = []
for categories, repos in categories_repos.items():
    for repo in repos:
        repos_list.append(repo)
repos_list = list(set(repos_list))
# print(len(repos_list)) # 427

# 5. Building repo-mean_embedding dict list
repo_mean_embedding_dict_list = []
for repo in repos_list:
    repo_mean_embedding_dict = {}
    repo_mean_embedding_dict["name"] = repo
    repo_mean_embedding_dict["topics"] = repo_embeddings_dict[repo]["topic"]
    repo_mean_embedding_dict["mean_code_embedding"] = get_mean_embedding(
        repo_embeddings_dict[repo]["codes_embeddings"])
    repo_mean_embedding_dict["mean_doc_embedding"] = get_mean_embedding(
        repo_embeddings_dict[repo]["docs_embeddings"])
    repo_mean_embedding_dict["mean_requirement_embedding"] = get_mean_embedding(
        repo_embeddings_dict[repo]["requirements_embeddings"])
    repo_mean_embedding_dict["mean_readme_embedding"] = get_mean_embedding(
        repo_embeddings_dict[repo]["readme_embeddings"])
    repo_mean_embedding_dict["mean_repo_embedding"] = \
        torch.cat([
            repo_mean_embedding_dict["mean_code_embedding"],
            repo_mean_embedding_dict["mean_doc_embedding"],
            repo_mean_embedding_dict["mean_requirement_embedding"],
            repo_mean_embedding_dict["mean_readme_embedding"]
        ])
    repo_mean_embedding_dict_list.append(repo_mean_embedding_dict)

# print(len(repo_mean_embedding_dict_list)) # 427

# 6. Similarity calculation
rows_list = []
for info1, info2 in tqdm(combinations(repo_mean_embedding_dict_list, 2)):
    rows_list.append(
        {
            "repo1": info1["name"],
            "repo2": info2["name"],
            "topics1": info1["topics"],
            "topics2": info2["topics"],
            "code_sim": cossim(
                info1["mean_code_embedding"], info2["mean_code_embedding"]
            ),
            "doc_sim": cossim(
                info1["mean_doc_embedding"], info2["mean_doc_embedding"]
            ),
            "requirement_sim": cossim(
                info1["mean_requirement_embedding"], info2["mean_requirement_embedding"]
            ),
            "readme_sim": cossim(
                info1["mean_readme_embedding"], info2["mean_readme_embedding"]
            ),
            "repo_sim": cossim(
                info1["mean_repo_embedding"], info2["mean_repo_embedding"]
            ),
        }
    )

# Saving the calculation result
df = pd.DataFrame(rows_list)
df = df.sort_values("repo_sim", ascending=False).reset_index(drop=True)
df.to_csv("evaluation_results_90951.csv", index=False)
print(f"[+] Evaluation results saved to evaluation_results_90951.csv!")
