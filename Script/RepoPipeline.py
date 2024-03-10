from typing import Dict, Any, List

import ast
import tarfile
import torch
import requests
import numpy as np
from ast import AsyncFunctionDef, ClassDef, FunctionDef, Module
from transformers import Pipeline
from tqdm.auto import tqdm


def extract_code_and_docs(text: str):
    """
    The method for extracting codes and docs in text.
    :param text: python file.
    :return: codes and docs set.
    """
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


def extract_readmes(file_content):
    """
    The method for extracting readmes.
    :param lines: readmes.
    :return: readme sentences.
    """
    readmes_set = set()
    lines = file_content.split('\n')
    for line in lines:
        line = line.replace("\n", "").strip()
        readmes_set.add(line)

    return readmes_set


def extract_requirements(file_content):
    """
    The method for extracting requirements.
    :param lines: requirements.
    :return: requirement libraries.
    """
    requirements_set = set()
    lines = file_content.split('\n')
    for line in lines:
        line = line.replace("\n", "").strip()
        try:
            if " == " in line:
                splitLine = line.split(" == ")
            else:
                splitLine = line.split("==")
            requirements_set.add(splitLine[0])
        except:
            pass

    return requirements_set


def get_metadata(repo_name, headers=None):
    """
    The method for getting metadata of repository from github_api.
    :param repo_name: repository name.
    :param headers: request headers.
    :return: response json.
    """
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
    """
    The method for extracting repositories information.
    :param repos: repositories.
    :param headers: request header.
    :return: a list for representing the information of each repository.
    """
    extracted_infos = []
    for repo_name in tqdm(repos, disable=len(repos) <= 1):
        # 1. Extracting metadata.
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

        # Download repo tarball bytes ---- Download repository.
        download_url = f"https://api.github.com/repos/{repo_name}/tarball"
        tqdm.write(f"[+] Downloading {repo_name}")
        try:
            response = requests.get(download_url, headers=headers, stream=True)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            tqdm.write(f"[-] Failed to download {repo_name}: {e}")
            continue

        # Extract repository files and parse them
        tqdm.write(f"[+] Extracting {repo_name} info")
        with tarfile.open(fileobj=response.raw, mode="r|gz") as tar:
            for member in tar:
                # 2. Extracting codes and docs.
                if member.name.endswith(".py") and member.isfile():
                    try:
                        file_content = tar.extractfile(member).read().decode("utf-8")
                        # extract_code_and_docs
                        code_set, docs_set = extract_code_and_docs(file_content)
                        repo_info["codes"].update(code_set)
                        repo_info["docs"].update(docs_set)
                    except UnicodeDecodeError as e:
                        tqdm.write(
                            f"[-] UnicodeDecodeError in {member.name}, skipping: \n{e}"
                        )
                    except SyntaxError as e:
                        tqdm.write(f"[-] SyntaxError in {member.name}, skipping: \n{e}")
                # 3. Extracting readme.
                elif (member.name.endswith("README.md") or member.name.endswith("README.rst")) and member.isfile():
                    try:
                        file_content = tar.extractfile(member).read().decode("utf-8")
                        # extract readme
                        readmes_set = extract_readmes(file_content)
                        repo_info["readmes"].update(readmes_set)
                    except UnicodeDecodeError as e:
                        tqdm.write(
                            f"[-] UnicodeDecodeError in {member.name}, skipping: \n{e}"
                        )
                    except SyntaxError as e:
                        tqdm.write(f"[-] SyntaxError in {member.name}, skipping: \n{e}")
                # 4. Extracting requirements.
                elif member.name.endswith("requirements.txt") and member.isfile():
                    try:
                        file_content = tar.extractfile(member).read().decode("utf-8")
                        # extract readme
                        requirements_set = extract_requirements(file_content)
                        repo_info["requirements"].update(requirements_set)
                    except UnicodeDecodeError as e:
                        tqdm.write(
                            f"[-] UnicodeDecodeError in {member.name}, skipping: \n{e}"
                        )
                    except SyntaxError as e:
                        tqdm.write(f"[-] SyntaxError in {member.name}, skipping: \n{e}")

        extracted_infos.append(repo_info)

    return extracted_infos


class RepoPipeline(Pipeline):
    """
    A custom pipeline for generating series of embeddings of a repository.
    """

    def __init__(self, github_token=None, *args, **kwargs):
        """
        The initial method for pipeline.
        :param github_token: github_token
        :param args: args
        :param kwargs: kwargs
        """
        super().__init__(*args, **kwargs)

        # Getting github token
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
        """
        The method for splitting parameters.
        :param pipeline_parameters: parameters
        :return: different parameters of different periods.
        """
        # The parameters of "preprocess" period.
        preprocess_parameters = {}
        if "github_token" in pipeline_parameters:
            preprocess_parameters["github_token"] = pipeline_parameters["github_token"]

        # The parameters of "forward" period.
        forward_parameters = {}
        if "max_length" in pipeline_parameters:
            forward_parameters["max_length"] = pipeline_parameters["max_length"]

        # The parameters of "postprocess" period.
        postprocess_parameters = {}
        return preprocess_parameters, forward_parameters, postprocess_parameters

    def preprocess(self, input_: Any, github_token=None) -> List:
        """
        The method for "preprocess" period.
        :param input_: the input.
        :param github_token: github_token.
        :return: a list about repository information.
        """
        # Making input to list format.
        if isinstance(input_, str):
            input_ = [input_]

        # Building headers.
        headers = {"Accept": "application/vnd.github+json"}
        token = github_token or self.github_token
        if token:
            headers["Authorization"] = f"Bearer {token}"

        # Getting repositories' information: input_ means series of repositories (can be only one repository).
        extracted_infos = extract_information(input_, headers=headers)
        return extracted_infos

    def encode(self, text, max_length):
        """
        The method for encoding the text to embedding by using UniXcoder.
        :param text: text.
        :param max_length: the max length.
        :return: the embedding of text.
        """
        assert max_length < 1024

        # Getting the tokenizer.
        tokenizer = self.tokenizer
        tokens = (
                [tokenizer.cls_token, "<encoder-only>", tokenizer.sep_token]
                + tokenizer.tokenize(text)[: max_length - 4]
                + [tokenizer.sep_token]
        )
        tokens_id = tokenizer.convert_tokens_to_ids(tokens)
        source_ids = torch.tensor([tokens_id]).to(self.device)
        token_embeddings = self.model(source_ids)[0]

        # Getting the text embedding.
        sentence_embeddings = token_embeddings.mean(dim=1)

        return sentence_embeddings

    def generate_embeddings(self, text_sets, max_length):
        """
        The method for generating embeddings of a text set.
        :param text_sets: text set.
        :param max_length: max length.
        :return: the embeddings of text set.
        """
        assert max_length < 1024

        # Concat the embeddings of each sentence/text in vertical dimension.
        return torch.zeros((1, 768), device=self.device) \
            if not text_sets \
            else torch.cat([self.encode(text, max_length) for text in text_sets], dim=0)

    def _forward(self, extracted_infos: List, max_length=512, st_progress=None) -> List:
        """
        The method for "forward" period.
        :param extracted_infos: the information of repositories.
        :param max_length: max length.
        :return: the output of this pipeline.
        """
        model_outputs = []
        # The number of repository.
        num_texts = sum(
            len(x["codes"]) + len(x["docs"]) + len(x["requirements"]) + len(x["readmes"]) for x in extracted_infos)
        with tqdm(total=num_texts) as progress_bar:
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
                info["mean_code_embedding"] = torch.mean(code_embeddings, dim=0, keepdim=True).cpu().numpy()
                progress_bar.update(len(repo_info["codes"]))
                if st_progress:
                    st_progress.progress(progress_bar.n / progress_bar.total)

                # Doc embeddings
                tqdm.write(f"[*] Generating doc embeddings for {repo_name}")
                doc_embeddings = self.generate_embeddings(repo_info["docs"], max_length)
                info["doc_embeddings"] = doc_embeddings.cpu().numpy()
                info["mean_doc_embedding"] = torch.mean(doc_embeddings, dim=0, keepdim=True).cpu().numpy()
                progress_bar.update(len(repo_info["docs"]))
                if st_progress:
                    st_progress.progress(progress_bar.n / progress_bar.total)

                # Requirement embeddings
                tqdm.write(f"[*] Generating requirement embeddings for {repo_name}")
                requirement_embeddings = self.generate_embeddings(repo_info["requirements"], max_length)
                info["requirement_embeddings"] = requirement_embeddings.cpu().numpy()
                info["mean_requirement_embedding"] = torch.mean(requirement_embeddings, dim=0,
                                                                keepdim=True).cpu().numpy()
                progress_bar.update(len(repo_info["requirements"]))
                if st_progress:
                    st_progress.progress(progress_bar.n / progress_bar.total)

                # Readme embeddings
                tqdm.write(f"[*] Generating readme embeddings for {repo_name}")
                readme_embeddings = self.generate_embeddings(repo_info["readmes"], max_length)
                info["readme_embeddings"] = readme_embeddings.cpu().numpy()
                info["mean_readme_embedding"] = torch.mean(readme_embeddings, dim=0, keepdim=True).cpu().numpy()
                progress_bar.update(len(repo_info["readmes"]))
                if st_progress:
                    st_progress.progress(progress_bar.n / progress_bar.total)

                # Repo-level mean embedding
                info["mean_repo_embedding"] = np.concatenate([
                    info["mean_code_embedding"],
                    info["mean_doc_embedding"],
                    info["mean_requirement_embedding"],
                    info["mean_readme_embedding"]
                ], axis=0).reshape(1, -1)

                info["code_embeddings_shape"] = info["code_embeddings"].shape
                info["mean_code_embedding_shape"] = info["mean_code_embedding"].shape
                info["doc_embeddings_shape"] = info["doc_embeddings"].shape
                info["mean_doc_embedding_shape"] = info["mean_doc_embedding"].shape
                info["requirement_embeddings_shape"] = info["requirement_embeddings"].shape
                info["mean_requirement_embedding_shape"] = info["mean_requirement_embedding"].shape
                info["readme_embeddings_shape"] = info["readme_embeddings"].shape
                info["mean_readme_embedding_shape"] = info["mean_readme_embedding"].shape
                info["mean_repo_embedding_shape"] = info["mean_repo_embedding"].shape

                model_outputs.append(info)

        return model_outputs

    def postprocess(self, model_outputs: List, **postprocess_parameters: Dict) -> List:
        """
        The method for "postprocess" period.
        :param model_outputs: the output of this pipeline.
        :param postprocess_parameters: the parameters of "postprocess" period.
        :return: model output.
        """
        return model_outputs
