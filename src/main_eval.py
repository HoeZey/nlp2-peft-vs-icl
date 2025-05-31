import json
import argparse
from pathlib import Path
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from datasets import load_dataset
from data import AIMEProcessor, GSM8KProcessor, MATHProcessor
from evaluation import evaluate, RandomFewshotSampler, SimilarityFewshotSampler
from hf import hf_login


def parse_args():
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--system_prompt_path", type=str, default=None)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--num_gpus", type=int, default=1)

    # Datasets
    parser.add_argument("--eval_dataset", type=str)
    parser.add_argument("--eval_dataset_config_name", type=str)
    parser.add_argument("--eval_split", type=str, default="test")
    parser.add_argument("--icl_dataset", type=str, default=None)
    parser.add_argument("--icl_dataset_config_name", type=str, default=None)
    parser.add_argument("--icl_split", type=str, default="train")
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument(
        "--sampling_strategy",
        type=str,
        default="random",
        choices=["random", "similarity"],
    )
    parser.add_argument(
        "--embedder", type=str, default="math-similarity/Bert-MLM_arXiv-MP-class_zbMath"
    )

    # Generation
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    # parser.add_argument("--num_workers", type=int, default=None)
    # parser.add_argument("--output_dir", type=str, default="./outputs/evaluation")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--output_dir", type=str)

    return parser.parse_args()


def main():
    hf_login()
    args = parse_args()
    print(args)

    eval_dataset = load_dataset(
        args.eval_dataset, name=args.eval_dataset_config_name, split=args.eval_split
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

    sampler = None
    sampling = ""
    if args.k > 0:
        icl_dataset = load_dataset(
            args.icl_dataset, name=args.icl_dataset_config_name, split=args.icl_split
        )
        if args.sampling_strategy == "random":
            sampler = RandomFewshotSampler(icl_dataset, k=args.k, seed=args.seed)
            sampling = "_r"
        elif args.sampling_strategy == "similarity":
            sampler = SimilarityFewshotSampler(
                icl_dataset, processor, model=args.embedder, k=args.k, seed=args.seed
            )
            sampling = "_s"
        else:
            raise ValueError(f"Unknown sampling strategy: {args.sampling_strategy}")

    if args.system_prompt_path is not None:
        with open(args.system_prompt_path, "r") as f:
            system_prompt = "\n".join(f.readlines())
    else:
        system_prompt = ""

    output_dir = Path(args.output_dir)
    answers_file = (
        f"{'_'.join(args.model_path.split('/'))}_k={args.k}{sampling}_answers.json"
    )
    output_file = (
        f"{'_'.join(args.model_path.split('/'))}_k={args.k}{sampling}_output.json"
    )

    if not (output_dir / answers_file).exists():
        output_dir.mkdir(exist_ok=True, parents=True)
        json.dump({}, (output_dir / answers_file).open("w"))

    if not (output_dir / output_file).exists():
        output_dir.mkdir(exist_ok=True, parents=True)
        json.dump({}, (output_dir / output_file).open("w"))

    model = LLM(args.model_path, tensor_parallel_size=args.num_gpus)
    sampling_params = SamplingParams(
        n=args.n,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stop_token_ids=[
            model.get_tokenizer().eos_token_id,
            model.get_tokenizer().convert_tokens_to_ids("<|eot_id|>"),
        ],
    )
    lora_request = LoRARequest(args.lora_path) if args.lora_path else None

    evaluate(
        model,
        sampling_params,
        eval_dataset=eval_dataset,
        processor=processor,
        batch_size=args.batch_size,
        answers_file=output_dir / answers_file,
        output_file=output_dir / output_file,
        fewshot_sampler=sampler,
        lora_request=lora_request,
        system_prompt=system_prompt,
    )


if __name__ == "__main__":
    main()
