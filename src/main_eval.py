import argparse
import torch
from torch.utils.data import DataLoader, RandomSampler
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from evaluation.evaluate import evaluate
from data import DataLoaderParams
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--lora_path", type=str)

    # Datasets
    parser.add_argument("--eval_datset", type=str)
    parser.add_argument("--eval_split", type=str, default="train")
    parser.add_argument("--icl_datset", type=str, default=None)
    parser.add_argument("--icl_split", type=str, default="train")

    # Generation
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_tokens", type=int, default=5000)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="./outputs/evaluation")
    # parser.add_argument("--eos_token_id", type=int, default=49152)

    return parser.parse_args()


def main():
    args = parse_args()
    model = LLM()
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
    )
    lora_request = LoRARequest()
    eval_dataset = load_dataset(args.eval_dataset, split=args.eval_split)

    if args.k > 0:
        icl_dataset = load_dataset(args.icl_dataset, split=args.icl_split)

    evaluate(
        model,
        sampling_params,
        lora_request,
        eval_dataset=eval_dataset,
        dataloader_params=DataLoaderParams(
            batch_size=args.batch_size,
            shuffle=False,
        ),
        k=args.k,
        icl_dataset=icl_dataset,
    )


if __name__ == "__main__":
    main()
