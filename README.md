# RepoSim4Py
A project for determining the similarity of python repositories based on embedding approach

## Requirements
```
accelerate==0.21.0
torch==2.0.1
numpy==1.25.0
pandas==2.0.3
transformers==4.30.2
sentence-transformers==2.2.2
tqdm==4.65.0
inspect4py==0.0.8
scikit-learn==1.3.0
matplotlib==3.7.2
```

## Getting started
```bash
pip install -r requirements.txt
```

```bash
python RepoSim4Py.py -i lepture/authlib idan/oauthlib evonove/django-oauth-toolkit selwin/python-user-agents SmileyChris/django-countries django-compressor/django-compressor billpmurphy/hask pytoolz/toolz Suor/funcy przemyslawjanpietrzak/pyMonet -o output/ --eval
```