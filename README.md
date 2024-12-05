<img src="https://cdn-avatars.huggingface.co/v1/production/uploads/634e20a0c1ce28f1de920cc4/k7SJny1M3lDa5CH_T1bp3.png" width="125" height="125" align="right" />

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
