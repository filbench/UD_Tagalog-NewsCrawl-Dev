import argparse
from pathlib import Path
from datasets import load_dataset
from collections import Counter
import pandas as pd


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset_name", type=str, default="UD-Filipino/UD_Tagalog-NewsCrawl", help="Dataset name containing the treebank.")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()
    dataset = load_dataset(args.dataset_name)

    def _get_counts(lst: list[str]) -> pd.DataFrame:
        items = [item for sublist in lst for item in sublist]
        ctr = Counter(items)
        df = pd.DataFrame.from_dict([ctr]).transpose()
        df = df[df.index != "_"]
        return df

    upos_tags = []
    deprels = []
    for split in ("train", "validation", "test"):
        df = dataset[split].to_pandas()
        upos_tags.append(_get_counts(df["upos_tags"].to_list()))
        deprels.append(_get_counts(df["deprels"].to_list()))

    upos_df = pd.concat(upos_tags, axis=1, join="outer").fillna(0)
    deprel_df = pd.concat(deprels, axis=1, join="outer").fillna(0)
    # Fix column names
    upos_df.columns = ["Train", "Dev", "Test"]
    deprel_df.columns = ["Train", "Dev", "Test"]
    # Sort by train size
    upos_df = upos_df.sort_values(by="Train", ascending=False)
    deprel_df = deprel_df.sort_values(by="Train", ascending=False)

    print(upos_df.to_latex(float_format="%d"))
    print("\n\n\n\n")
    print(deprel_df.to_latex(float_format="%d"))


if __name__ == "__main__":
    main()
