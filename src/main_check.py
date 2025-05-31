import json
import argparse
from pathlib import Path
from datasets import load_dataset
from data import AIMEProcessor, GSM8KProcessor, MATHProcessor
from evaluation import check_answers
from hf import hf_login


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str)
    parser.add_argument("--eval_dataset", type=str)
    parser.add_argument("--eval_dataset_config_name", type=str)
    parser.add_argument("--eval_split", type=str, default="test")
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument(
        "--sampling_strategy",
        type=str,
        default="random",
        choices=["random", "similarity"],
    )

    return parser.parse_args()


def main():
    hf_login()
    args = parse_args()
    print(args)

    eval_dataset = load_dataset(
        args.eval_dataset, name=args.eval_dataset_config_name, split=args.eval_split
    )
    if args.k > 0:
        sampling = "_r" if args.sampling_strategy == "random" else "_s"
    else:
        sampling = ""
    output_dir = Path(args.output_dir)
    answers_file = (
        f"{'_'.join(args.model_path.split('/'))}_k={args.k}{sampling}_answers.json"
    )
    if "gsm8k" in args.eval_dataset.lower():
        processor = GSM8KProcessor()
        print("Using GSM8K processor")
    elif "math" in args.eval_dataset.lower():
        processor = MATHProcessor()
        print("Using MATH processor")
    elif "aime" in args.eval_dataset.lower():
        processor = AIMEProcessor()
        print("Using AIME processor")
    else:
        raise ValueError(f"Unknown dataset: {args.eval_dataset}")

    if not (output_dir / answers_file).exists():
        output_dir.mkdir(exist_ok=True, parents=True)
        json.dump({}, (output_dir / answers_file).open("w"))

    check_answers(eval_dataset, processor, output_dir / answers_file)


if __name__ == "__main__":
    main()
