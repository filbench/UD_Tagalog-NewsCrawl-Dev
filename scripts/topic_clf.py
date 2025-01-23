import argparse
import logging
from pathlib import Path

from datasets import load_dataset
from vllm import LLM, SamplingParams, RequestOutput


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
    logging.info(f"Downloading dataset {args.dataset_name}")
    texts = []
    for split in ("train", "test", "validation"):
        df = load_dataset(args.dataset_name, split=split).to_pandas()
        texts.extend(df["text"].to_list())

    prompts = [format_prompt(text) for text in texts]

    # Set-up vLLM
    llm = LLM(
        model=args.model_name,
        tokenizer=args.model_name,
        dtype="auto",
        tokenizer_mode="auto",
        max_num_seqs=args.batch_size,
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


if __name__ == "__main__":
    main()
