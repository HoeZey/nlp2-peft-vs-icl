import argparse
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from datasets import load_dataset
from dotenv import load_dotenv
from evaluation.evaluate import evaluate, FewshotSampler
from hf import hf_login


def parse_args():
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--system_prompt_path", type=str)
    parser.add_argument("--lora_path", type=str, default=None)

    # Datasets
    parser.add_argument("--eval_dataset", type=str)
    parser.add_argument("--eval_split", type=str, default="test")
    parser.add_argument("--icl_datset", type=str, default=None)
    parser.add_argument("--icl_split", type=str, default="train")
    parser.add_argument("--k", type=int, default=0)

    # Generation
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_tokens", type=int, default=5000)
    # parser.add_argument("--num_workers", type=int, default=None)
    # parser.add_argument("--output_dir", type=str, default="./outputs/evaluation")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    hf_login()
    args = parse_args()
    model = LLM(args.model_path)
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
    )
    lora_request = LoRARequest(args.lora_path) if args.lora_path else None
    eval_dataset = load_dataset(args.eval_dataset, split=args.eval_split)

    sampler = None
    if args.k > 0:
        icl_dataset = load_dataset(args.icl_dataset, split=args.icl_split)
        sampler = FewshotSampler(icl_dataset, k=args.k, seed=args.seed)

    with open(args.system_prompt_path, "r") as f:
        system_prompt = "\n".join(f.readlines())

    evaluate(
        model,
        sampling_params,
        eval_dataset=eval_dataset,
        batch_size=args.batch_size,
        fewshot_sampler=sampler,
        lora_request=lora_request,
        system_prompt=system_prompt,
    )


if __name__ == "__main__":
    main()
