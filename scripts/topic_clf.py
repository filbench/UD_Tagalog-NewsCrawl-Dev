import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
from datasets import load_dataset
from vllm import LLM, SamplingParams, RequestOutput

FONT_SIZES = {"small": 14, "medium": 18, "large": 24}

PLOT_PARAMS = {
    "font.family": "serif",
    "font.serif": ["Times"],
    "font.size": FONT_SIZES.get("medium"),
    "axes.titlesize": FONT_SIZES.get("large"),
    "axes.labelsize": FONT_SIZES.get("large"),
    "xtick.labelsize": FONT_SIZES.get("large"),
    "ytick.labelsize": FONT_SIZES.get("large"),
    "legend.fontsize": FONT_SIZES.get("medium"),
    "figure.titlesize": FONT_SIZES.get("medium"),
    "text.usetex": True,
}

plt.rcParams.update(PLOT_PARAMS)


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    subparsers = parser.add_subparsers(dest="command")

    # fmt: off
    infer = subparsers.add_parser("infer", help="Perform topic classification")
    infer.add_argument("--dataset_name", type=str, default="UD-Filipino/UD_Tagalog-NewsCrawl")
    infer.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    infer.add_argument("--batch_size", type=int, default=128, help="Set the batch size.")
    infer.add_argument("--output_path", type=Path, default="topics.jsonl", help="Path to save the outputs.")

    plot = subparsers.add_parser("plot", help="Plot results.")
    plot.add_argument("--output_path", type=Path, default="topic_clf.pdf", help="Path to save the PDF plot")
    plot.add_argument("--figsize", type=int, nargs=2, default=[10, 10], help="Matplotlib figure size.")

    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()
    if args.command == "infer":
        infer(
            dataset_name=args.dataset_name,
            model_name=args.model_name,
            batch_size=args.batch_size,
            output_path=args.output_path,
        )
    elif args.command == "plot":
        plot(output_path=args.output_path, figsize=args.figsize)
    else:
        logging.error(f"Unknown command: '{args.command}'")
        raise


def infer(dataset_name: str, model_name: str, output_path: Path, batch_size: int):
    logging.info(f"Downloading dataset {args.dataset_name}")
    texts = []
    for split in ("train", "test", "validation"):
        df = load_dataset(dataset_name, split=split).to_pandas()
        texts.extend(df["text"].to_list())

    prompts = [format_prompt(text) for text in texts]

    # Set-up vLLM
    llm = LLM(
        model=model_name,
        tokenizer=model_name,
        dtype="auto",
        tokenizer_mode="auto",
        max_num_seqs=batch_size,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(n=1, temperature=0.1, max_tokens=100)
    responses: list[RequestOutput] = llm.generate(
        prompts=prompts,
        sampling_params=sampling_params,
        use_tqdm=True,
    )

    # Parse the raw responses
    outputs: list[dict[str, str]] = []
    for response in responses:
        outputs.append({"id": response.request_id, "output": response.outputs[0].text})

    # Save outputs into the specified output path
    output_path = args.output_path
    with open(output_path, "w", encoding="utf-8") as f:
        for output in outputs:
            f.write(f"{output}\n")
    logging.info(f"Outputs saved to {output_path}")


def format_prompt(text: str) -> str:
    template = """Given the text below, classify it into one of the following topics:

science/technology
travel
politics
sports
health
entertainment
geography
unknown

The text is in Tagalog, but I want your response to be in English.
Try not to predict the 'unknown' category, use it if you really don't know the topic.
Return only the topic name in small caps. 

Below are some examples:

Text: Sa simula ng digmaan kalimitang naglakbay ang mga ito sa ibabaw ng dagat, nguni't noong nagsimula nang sumulong ang radar at nagiging mas tumpak na ito, ang mga submarino ay napilitang sumisid sa ilalim ng tubig upang maiwasang may makakita sa mga ito.
Answer: science/technology

Text: Noong 1994, humantong ang di-pagkakasundo sa paglikha ng nag-aangking Republika ng Transnitria sa silangang Moldova, na may sariling pamahalaan at pera subalit hindi kinikilala ng anumang miyembrong bansa ng UN. 
Answer: politics

Here is the text you need to classify:

Text: {text}
Answer:
"""
    return template.format(text=text)


def plot(output_path: Path, figsize: tuple[int, int]):
    breakpoint()
    categories = [
        "entertainment",
        "health",
        "sports",
        "politics",
        "travel",
        "geography",
        "science/technology",
    ]
    percentages = [
        36.688573,
        25.637071,
        17.509388,
        13.324839,
        3.192060,
        2.829936,
        0.818133,
    ]


if __name__ == "__main__":
    main()
