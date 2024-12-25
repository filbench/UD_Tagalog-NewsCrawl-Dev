import os
import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import conllu
import pandas as pd
from conllu import parse_incr
from datasets import Dataset, Split, DatasetDict
from spacy.morphology import Morphology

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)

NONE_IDENTIFIER = "_"


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_dir", type=Path, default="assets/", help="Input directory containing all the files.")
    parser.add_argument("--dataset", type=str, default="UD-Filipino/UD_Tagalog-NewsCrawl")
    parser.add_argument("--dry_run", action="store_true", default=False, help="If set, will process the dataset but will not upload to HuggingFace.")
    # fmt: on
    return parser.parse_args()


def main():
    hf_token = os.getenv("HF_TOKEN", None)
    if not hf_token:
        raise ValueError(
            "HF_TOKEN not set! Please generate a token from https://huggingface.co/settings/tokens"
        )

    args = get_args()
    input_dir = Path(args.input_dir)
    filename = "tl_newscrawl-ud"

    split2file = {
        Split.TRAIN: input_dir / f"{filename}-train.conllu",
        Split.VALIDATION: input_dir / f"{filename}-dev.conllu",
        Split.TEST: input_dir / f"{filename}-test.conllu",
    }

    # Basically your goal is to create a DatasetDict
    dataset_dict = DatasetDict()
    for split_name, path in split2file.items():
        with path.open("r", encoding="utf-8") as f:
            sents = list(parse_incr(f))
        logging.info(f"Loaded split '{split_name}' from {path} ({len(sents)} sents)")

        examples = []
        for sent in sents:
            example = create_example(sent)
            examples.append(example)

        df = pd.DataFrame.from_dict(examples)
        dataset = Dataset.from_dict(df)
        dataset_dict[str(split_name)] = dataset

    print(dataset_dict)
    if not args.dry_run:
        dataset_dict.push_to_hub(args.dataset)


def create_example(sent: conllu.TokenList) -> dict[str, Any]:
    id = sent.metadata["sent_id"]
    text = sent.metadata["text"]

    tokens: list[str] = []
    lemmas: list[str] = []
    xpos_tags: list[str] = []
    upos_tags: list[str] = []
    feats: list[str] = []
    heads: list[int] = []
    deprels: list[str] = []
    for token in sent:
        tokens.append(token.get("form"))
        lemmas.append(token.get("lemma"))
        xpos_tags.append(token.get("xpos") if token.get("xpos") else NONE_IDENTIFIER)
        upos_tags.append(token.get("upos") if token.get("upos") else NONE_IDENTIFIER)
        feats.append(
            Morphology.dict_to_feats(token.get("feats"))
            if token.get("feats")
            else NONE_IDENTIFIER
        )
        heads.append(token.get("head", ""))
        deprels.append(token.get("deprel"))

    example = {
        "id": id,
        "text": text,
        "tokens": tokens,
        "lemmas": lemmas,
        "xpos_tags": xpos_tags,
        "upos_tags": upos_tags,
        "feats": feats,
        "heads": heads,
        "deprels": deprels,
    }
    return example


if __name__ == "__main__":
    main()
