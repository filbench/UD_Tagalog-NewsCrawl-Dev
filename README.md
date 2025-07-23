<img src="https://cdn-avatars.huggingface.co/v1/production/uploads/634e20a0c1ce28f1de920cc4/k7SJny1M3lDa5CH_T1bp3.png" width="130" height="130" align="right" />

# UD_Tagalog-NewsCrawl (Dev)

<p align="left">
<b><a href="https://huggingface.co/datasets/UD-Filipino/UD_Tagalog-NewsCrawl">🤗 Dataset</a></b>
|
<b><a href="https://huggingface.co/collections/UD-Filipino/universal-dependencies-for-tagalog-67573d625baa5036fd59b317">🤗 Baselines</a></b>
|
<b><a href="https://arxiv.org/abs/2505.20428">📄 Paper</a></b>
</p>

This repository contains several experiments and benchmarks for the ACL 2025 (Main) paper [*The UD-NewsCrawl Treebank: Reflections and Challenges from a
Large-scale Tagalog Syntactic Annotation Project*](https://arxiv.org/abs/2505.20428) which introduces [UD_Tagalog-NewsCrawl](https://huggingface.co/datasets/UD-Filipino/UD_Tagalog-NewsCrawl), the largest Tagalog treebank to date.

## News

- [2025-05-15] The UD-NewsCrawl Treebank was accepted at the ACL 2025 (Main) Conference!

## Set-up

To get started, install all dependencies in a virtual environment and download all assets:

```sh
python3 -m venv venv
source venv/bin/activate
# Within the virtual environment
python3 -m pip install -r requirements.txt
python3 -m spacy project assets
```

This will create an `assets/` directory and contain important data files, especially the train, dev, and test splits for UD-NewsCrawl.
Note that the dataset is also available in [HuggingFace](https://huggingface.co/datasets/UD-Filipino/UD_Tagalog-NewsCrawl) (but for our purposes we'll use the orig files):

```python
# pip install datasets
from datasets import load_dataset
ds = load_dataset("UD-Filipino/UD_Tagalog-NewsCrawl")
```

## Training the models

We use the [spaCy projects](https://spacy.io/usage/projects) framework to manage end-to-end training workflows.
You can train a model by passing a model ID (see table below) later to the following command:

```sh
spacy project run setup  # run this once
spacy project run <MODEL-ID>
```

There are six (5) models as outlined in our paper. They are also available in this [HuggingFace collection](https://huggingface.co/collections/UD-Filipino/universal-dependencies-for-tagalog-67573d625baa5036fd59b317):

| ID                  | Description                                                                                            |
|---------------------|--------------------------------------------------------------------------------------------------------|
| baseline-transition | **no word embeddings** on a transition-based parser based on spaCy.                                  |
| fasttext-transition | fastText word embeddings on a transition-based parser based on spaCy.                                  |
| hash-transition     | [Multi-hash embeddings](https://arxiv.org/abs/2212.09255) on a transition-based parser based on spaCy. |
| xling-transition    | XLM-RoBERTa context-sensitive vectors on a transition-based parser based on spaCy.                     |
| mono-transition     | RoBERTa-Tagalog context-sensitive vectors on a transition-based parser based on spaCy.                 |

## Cite

If you're using the UD-NewsCrawl treebank for your project, please cite:

```
@inproceedings{aquino-etal-2025-ud,
    title = "The {UD}-{N}ews{C}rawl Treebank: Reflections and Challenges from a Large-scale {T}agalog Syntactic Annotation Project",
    author = "Aquino, Angelina Aspra  and
      Miranda, Lester James Validad  and
      Or, Elsie Marie T.",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.357/",
    pages = "7219--7239",
    ISBN = "979-8-89176-251-0",
    abstract = "This paper presents UD-NewsCrawl, the largest Tagalog treebank to date, containing 15.6k trees manually annotated according tothe Universal Dependencies framework. We detail our treebank development process, including data collection, pre-processing, manual annotation, and quality assurance procedures. We provide baseline evaluations using multiple transformer-based models to assess the performance of state-of-the-art dependency parsers on Tagalog. We also highlight challenges in the syntactic analysis of Tagalog given its distinctive grammatical properties, and discuss its implications for the annotation of this treebank. We anticipate that UD-NewsCrawl and our baseline model implementations will serve as valuable resources for advancing computational linguistics research in underrepresented languages like Tagalog."
}
```
