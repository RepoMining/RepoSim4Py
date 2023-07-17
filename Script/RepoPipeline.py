from typing import Dict, Any, List

import ast
import tarfile
from ast import AsyncFunctionDef, ClassDef, FunctionDef, Module
import torch
import requests
from transformers import Pipeline
from tqdm.auto import tqdm


def extract_code_and_docs(text: str):
    code_set = set()
    docs_set = set()
    root = ast.parse(text)
    for node in ast.walk(root):
        if not isinstance(node, (AsyncFunctionDef, FunctionDef, ClassDef, Module)):
            continue
        docs = ast.get_docstring(node)
        node_without_docs = node
        if docs is not None:
            docs_set.add(docs)
            # Remove docstrings from the node
            node_without_docs.body = node_without_docs.body[1:]
        if isinstance(node, (AsyncFunctionDef, FunctionDef)):
            code_set.add(ast.unparse(node_without_docs))

    return code_set, docs_set


def get_metadata(repo_name, headers=None):
    api_url = f"https://api.github.com/repos/{repo_name}"
    tqdm.write(f"[+] Getting metadata for {repo_name}")
    try:
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        tqdm.write(f"[-] Failed to retrieve metadata from {repo_name}: {e}")
        return {}


def extract_information(repos, headers=None):
    extracted_infos = []
    for repo_name in tqdm(repos, disable=len(repos) <= 1):
        # Get metadata
        metadata = get_metadata(repo_name, headers=headers)
        repo_info = {
            "name": repo_name,
            "codes": set(),
            "docs": set(),
            "requirements": set(),
            "readmes": set(),
            "topics": [],
            "license": "",
            "stars": metadata.get("stargazers_count"),
        }
        if metadata.get("topics"):
            repo_info["topics"] = metadata["topics"]
        if metadata.get("license"):
            repo_info["license"] = metadata["license"]["spdx_id"]

        # Download repo tarball bytes
        download_url = f"https://api.github.com/repos/{repo_name}/tarball"
        tqdm.write(f"[+] Downloading {repo_name}")
        try:
            response = requests.get(download_url, headers=headers, stream=True)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            tqdm.write(f"[-] Failed to download {repo_name}: {e}")
            continue

        # Extract python files and parse them
        tqdm.write(f"[+] Extracting {repo_name} info")
        with tarfile.open(fileobj=response.raw, mode="r|gz") as tar:
            for member in tar:
                if (member.name.endswith(".py") and member.isfile()) is False:
                    continue
                try:
                    file_content = tar.extractfile(member).read().decode("utf-8")
                    code_set, docs_set = extract_code_and_docs(file_content)

                    repo_info["codes"].update(code_set)
                    repo_info["docs"].update(docs_set)
                except UnicodeDecodeError as e:
                    tqdm.write(
                        f"[-] UnicodeDecodeError in {member.name}, skipping: \n{e}"
                    )
                except SyntaxError as e:
                    tqdm.write(f"[-] SyntaxError in {member.name}, skipping: \n{e}")

        extracted_infos.append(repo_info)

    return extracted_infos


class RepoPipeline(Pipeline):

    def __init__(self, github_token=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Github token
        self.github_token = github_token
        if self.github_token:
            print("[+] GitHub token set!")
        else:
            print(
                "[*] Please set GitHub token to avoid unexpected errors. \n"
                "For more info, see: "
                "https://docs.github.com/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token"
            )

    def _sanitize_parameters(self, **pipeline_parameters):
        preprocess_parameters = {}
        if "github_token" in pipeline_parameters:
            preprocess_parameters["github_token"] = pipeline_parameters["github_token"]

        forward_parameters = {}
        if "max_length" in pipeline_parameters:
            forward_parameters["max_length"] = pipeline_parameters["max_length"]

        postprocess_parameters = {}
        return preprocess_parameters, forward_parameters, postprocess_parameters

    def preprocess(self, input_: Any, github_token=None) -> List:
        # Making input to list format
        if isinstance(input_, str):
            input_ = [input_]

        # Building token
        headers = {"Accept": "application/vnd.github+json"}
        token = github_token or self.github_token
        if token:
            headers["Authorization"] = f"Bearer {token}"

        # Getting repositories' information: input_ means series of repositories
        extracted_infos = extract_information(input_, headers=headers)

        return extracted_infos

    def encode(self, text, max_length):
        assert max_length < 1024

        tokenizer = self.tokenizer
        tokens = (
                [tokenizer.cls_token, "<encoder-only>", tokenizer.sep_token]
                + tokenizer.tokenize(text)[: max_length - 4]
                + [tokenizer.sep_token]
        )
        tokens_id = tokenizer.convert_tokens_to_ids(tokens)
        source_ids = torch.tensor([tokens_id]).to(self.device)

        token_embeddings = self.model(source_ids)[0]
        sentence_embeddings = token_embeddings.mean(dim=1)

        return sentence_embeddings

    def generate_embeddings(self, text_sets, max_length):
        assert max_length < 1024
        return torch.zeros((1, 768), device=self.device) \
            if text_sets is None or len(text_sets) == 0 \
            else torch.cat([self.encode(text, max_length) for text in text_sets], dim=0)

    def _forward(self, extracted_infos: List, max_length=512) -> List:
        model_outputs = []
        num_repos = len(extracted_infos)
        with tqdm(total=num_repos) as progress_bar:
            # For each repository
            for repo_info in extracted_infos:
                repo_name = repo_info["name"]
                info = {
                    "name": repo_name,
                    "topics": repo_info["topics"],
                    "license": repo_info["license"],
                    "stars": repo_info["stars"],
                }
                progress_bar.set_description(f"Processing {repo_name}")

                # Code embeddings
                tqdm.write(f"[*] Generating code embeddings for {repo_name}")
                code_embeddings = self.generate_embeddings(repo_info["codes"], max_length)
                info["code_embeddings"] = code_embeddings.cpu().numpy()
                info["mean_code_embedding"] = torch.mean(code_embeddings, dim=0).cpu().numpy()

                # Doc embeddings
                tqdm.write(f"[*] Generating doc embeddings for {repo_name}")
                doc_embeddings = self.generate_embeddings(repo_info["docs"], max_length)
                info["doc_embeddings"] = doc_embeddings.cpu().numpy()
                info["mean_doc_embedding"] = torch.mean(doc_embeddings, dim=0).cpu().numpy()

                # Requirement embeddings
                tqdm.write(f"[*] Generating requirement embeddings for {repo_name}")
                requirement_embeddings = self.generate_embeddings(repo_info["requirements"], max_length)
                info["requirement_embeddings"] = requirement_embeddings.cpu().numpy()
                info["mean_requirement_embedding"] = torch.mean(requirement_embeddings, dim=0).cpu().numpy()

                # Requirement embeddings
                tqdm.write(f"[*] Generating readme embeddings for {repo_name}")
                readme_embeddings = self.generate_embeddings(repo_info["readmes"], max_length)
                info["readme_embeddings"] = readme_embeddings.cpu().numpy()
                info["mean_readme_embedding"] = torch.mean(readme_embeddings, dim=0).cpu().numpy()

                progress_bar.update(1)
                model_outputs.append(info)

        return model_outputs

    def postprocess(self, model_outputs: List, **postprocess_parameters: Dict) -> List:
        return model_outputs


