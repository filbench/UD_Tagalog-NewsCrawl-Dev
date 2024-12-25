import argparse
from pathlib import Path


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metrics_dir", type=Path, default="metrics", help="Path to the metrics directory")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()
    breakpoint()


if __name__ == "__main__":
    main()
