# UD_Tagalog-NewsCrawl (Dev)

This repository contains the experiments and benchmarks for the [UD_Tagalog-NewsCrawl treebank](https://huggingface.co/datasets/UD-Filipino/UD_Tagalog-NewsCrawl).

## Downloading the dataset

The dataset is available as a set of CoNLL-U files which can be downloaded directly from Universal Dependencies:

```sh
bash scripts/download_conllu.sh dev
```

You can also download it as a HuggingFace dataset:

```python
# pip install datasets
from datasets import load_dataset
ds = load_dataset("UD-Filipino/UD_Tagalog-NewsCrawl")
```
