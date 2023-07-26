"""
The python file for running this program.

Running example:
>> pip install -r requirements.txt
>> python RepoSim4Py.py -i lepture/authlib idan/oauthlib evonove/django-oauth-toolkit selwin/python-user-agents SmileyChris/django-countries django-compressor/django-compressor billpmurphy/hask pytoolz/toolz Suor/funcy przemyslawjanpietrzak/pyMonet -o output/ --eval
"""

import os
import argparse
import pickle
from itertools import combinations
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from torch.nn.functional import cosine_similarity
from transformers import pipeline


def cossim(embedding1: np.ndarray, embedding2: np.ndarray):
    """
    The method for calculating cosine similarity of two numpy array (two vectors).
    :param embedding1: array 1, embedding 1
    :param embedding2: array 2, embedding 2
    :return: the cosine similarity of these two vectors.
    """

    embedding1 = torch.tensor(embedding1, dtype=torch.float32).squeeze(0)
    embedding2 = torch.tensor(embedding2, dtype=torch.float32).squeeze(0)
    return cosine_similarity(embedding1, embedding2, dim=0).item()


def main():
    """
    The main method for running this program.
    :return: None
    """

    # Arguments rules
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-i",
        "--input",
        nargs="+",
        help="Input repositories",
        required=True,
    )
    parser.add_argument("-o", "--output", help="Output directory", required=True)
    parser.add_argument(
        "-e",
        "--eval",
        help="Evaluate cosine similarities between all repository combinations",
        action="store_true",
    )
    args = parser.parse_args()

    # Building the output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    # Loading model
    model = pipeline(
        model="Henry65/RepoSim4Py",
        trust_remote_code=True,
        device_map="auto",
        github_token=os.environ.get("GITHUB_TOKEN")
    )

    # Getting input
    REPOS = args.input

    # Generating output by model and saving it
    output = model(tuple(REPOS))
    with open(output_dir / "output.pkl", "wb") as f:
        pickle.dump(output, f)
        print(f"[+] Model outputs was saved to {output_dir}/output.pkl!")

    # Starting to process evaluation
    if not args.eval:
        return

    if len(REPOS) < 2:
        print("[-] At least 2 repositories can be compared for similarity!")
        return

    # Similarity calculation
    rows_list = []
    for info1, info2 in combinations(output, 2):
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
    df.to_csv(output_dir / "evaluation_result.csv", index=False)
    print(f"[+] Evaluation results saved to {output_dir}/evaluation_result.csv!")


if __name__ == "__main__":
    main()
