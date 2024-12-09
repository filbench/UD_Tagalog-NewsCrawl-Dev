<img src="https://cdn-avatars.huggingface.co/v1/production/uploads/634e20a0c1ce28f1de920cc4/k7SJny1M3lDa5CH_T1bp3.png" width="130" height="130" align="right" />

# UD_Tagalog-NewsCrawl (Dev)

This repository contains several experiments and benchmarks for the paper [_UD-NewsCrawl: A Universal Dependencies Treebank for Tagalog_]() which introduces [UD_Tagalog-NewsCrawl](https://huggingface.co/datasets/UD-Filipino/UD_Tagalog-NewsCrawl), the largest Tagalog treebank to date.

To get started, install all dependencies in a virtual environment:

```sh
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```

## Downloading the dataset

The dataset is available as a set of CoNLL-U files which can be downloaded directly from Universal Dependencies:

```sh
bash utils/download_conllu.sh dev
```

You can also download it as a HuggingFace dataset:

```python
# pip install datasets
from datasets import load_dataset
ds = load_dataset("UD-Filipino/UD_Tagalog-NewsCrawl")
```

## Training the models

We use the [spaCy projects](https://spacy.io/usage/projects) framework to manage end-to-end training workflows.
You can train a model by passing a model ID (we'll explain) later to the following command:

```sh
spacy project run <MODEL-ID>
```

There are six (6) models as outlined in our paper:

| ID                  | Description                                                                                            |
|---------------------|--------------------------------------------------------------------------------------------------------|
| fasttext-graph      | fastText word embeddings on a graph-based parser using UDPipe                                          |
| fasttext-transition | fastText word embeddings on a transition-based parser based on spaCy.                                  |
| hash-transition     | [Multi-hash embeddings](https://arxiv.org/abs/2212.09255) on a transition-based parser based on spaCy. |
| xling-graph         | XLM-RoBERTa context-sensitive vectors on a graph-based parser using UDPipe                             |
| xling-transition    | XLM-RoBERTa context-sensitive vectors on a transition-based parser based on spaCy.                     |
| mono-transition     | RoBERTa-Tagalog context-sensitive vectors on a transition-based parser based on spaCy.                 |
