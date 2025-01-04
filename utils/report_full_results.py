import argparse
import pandas as pd
import json
from pathlib import Path


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metrics_dir", type=Path, default="metrics", help="Path to the metrics directory")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()
    metrics_dir = Path(args.metrics_dir)
    result_files = list(metrics_dir.glob("*.json"))

    morph_scores = []
    dep_scores = []

    for result_file in result_files:
        name = result_file.stem
        with open(result_file, "r") as f:
            data = json.load(f)

        morph_score = (
            pd.DataFrame.from_dict(data.get("morphologizer").get("morph_per_feat"))
            .transpose()["f"]
            .transpose()
        ) * 100
        morph_score = morph_score.rename(name)
        morph_scores.append(morph_score)

        dep_score = (
            pd.DataFrame.from_dict(data.get("parser").get("dep_las_per_type"))
            .transpose()["f"]
            .transpose()
        ) * 100
        dep_score = dep_score.rename(name)
        dep_scores.append(dep_score)

    morph_full = update_cols(pd.concat(morph_scores, axis=1))
    print(morph_full.to_latex(float_format="%.2f"))
    print("\n" * 4)
    dep_full = update_cols(pd.concat(dep_scores, axis=1))
    print(dep_full.to_latex(float_format="%.2f"))

    breakpoint()


def update_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df[
        [
            "tl_spacy_baseline",
            "tl_fasttext_transition",
            "tl_hash_transition",
            "tl_mdeberta_v3_transition",
            "tl_roberta_tgl_transition",
            "tl_xlm_roberta_transition",
        ]
    ]

    df = df.rename(
        columns={
            "tl_spacy_baseline": "No embeddings",
            "tl_fasttext_transition": "fastText",
            "tl_hash_transition": "Multi hash embeddings",
            "tl_mdeberta_v3_transition": "mDeBERTa-v3, base",
            "tl_roberta_tgl_transition": "RoBERTa-Tagalog, large",
            "tl_xlm_roberta_transition": "XLM-RoBERTa, large",
        }
    )
    return df


if __name__ == "__main__":
    main()
