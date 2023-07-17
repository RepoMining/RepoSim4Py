from transformers import pipeline
import os

model = pipeline(model="Henry65/RepoSim4Py", trust_remote_code=True, device_map="auto", github_token=os.environ.get("GITHUB_TOKEN"))
repo_infos = model("lazyhope/python-hello-world")
print(repo_infos)
