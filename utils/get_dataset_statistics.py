import argparse
from pathlib import Path


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset_name", type=str, default="UD-Filipino/UD_Tagalog-NewsCrawl", help="Dataset name containing the treebank.")
    parser.add_argument("--output_file", type=Path, required=False, help="Path to save the metrics results.")
    # fmt: on
    pass


def main():
    args = get_args()
    output_file = Path(args.output_file)
    breakpoint()


if __name__ == "__main__":
    main()
