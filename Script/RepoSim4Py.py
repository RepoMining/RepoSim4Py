"""
The python file for running this program.
"""

from transformers import pipeline
import os
from RepoPipeline import RepoPipeline

model = pipeline(model="Henry65/RepoSim4Py", trust_remote_code=True, device_map="auto",
                 github_token=os.environ.get("GITHUB_TOKEN"))
repo_infos = model("SoftwareUnderstanding/inspect4py")
print(repo_infos)
