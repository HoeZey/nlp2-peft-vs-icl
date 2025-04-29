import json
import argparse
from pathlib import Path
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from datasets import load_dataset
from evaluation.evaluate import evaluate, FewshotSampler
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

    # Generation
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_tokens", type=int, default=5000)
    # parser.add_argument("--num_workers", type=int, default=None)
    # parser.add_argument("--output_dir", type=str, default="./outputs/evaluation")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--output_dir", type=str)

    return parser.parse_args()


def main():
    hf_login()
    args = parse_args()
    print(args)

    model = LLM(args.model_path, tensor_parallel_size=args.num_gpus)
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
    )
    lora_request = LoRARequest(args.lora_path) if args.lora_path else None
    eval_dataset = load_dataset(
        args.eval_dataset, name=args.eval_dataset_config_name, split=args.eval_split
    )

    sampler = None
    if args.k > 0:
        icl_dataset = load_dataset(
            args.icl_dataset, name=args.icl_dataset_config_name, split=args.icl_split
        )
        sampler = FewshotSampler(icl_dataset, k=args.k, seed=args.seed)

    if args.system_prompt_path is not None:
        with open(args.system_prompt_path, "r") as f:
            system_prompt = "\n".join(f.readlines())
    else:
        system_prompt = ""

    output_dir = Path(args.output_dir)
    answers_file = f"{'_'.join(args.model_path.split('/'))}_k={args.k}_answers.json"
    output_file = f"{'_'.join(args.model_path.split('/'))}_k={args.k}_output.json"

    if not (output_dir / answers_file).exists():
        output_dir.mkdir(exist_ok=True, parents=True)
        json.dump({}, (output_dir / answers_file).open("w"))

    if not (output_dir / output_file).exists():
        output_dir.mkdir(exist_ok=True, parents=True)
        json.dump({}, (output_dir / output_file).open("w"))

    evaluate(
        model,
        sampling_params,
        eval_dataset=eval_dataset,
        batch_size=args.batch_size,
        answers_file=output_dir / answers_file,
        output_file=output_dir / output_file,
        fewshot_sampler=sampler,
        lora_request=lora_request,
        system_prompt=system_prompt,
    )


if __name__ == "__main__":
    main()
