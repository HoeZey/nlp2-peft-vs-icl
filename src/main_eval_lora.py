import os
import json
from datasets import load_dataset
from argparse import ArgumentParser
from hf import hf_login
from lora.evaluate import evaluate_model, load_fine_tuned_model
from lora.utils import models


def main():
    hf_login()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = ArgumentParser()
    parser.add_argument(
        "--model_type",
        type=str,
        choices=models.keys(),
        required=True,
        help="Type of model to evaluate (e.g., llama2, llama3).",
    )
    parser.add_argument(
        "--peft_model_path",
        type=str,
        required=True,
        help="Path to the fine-tuned model",
    )
    parser.add_argument(
        "--gen_max_length", type=int, default=512, help="Max length for generation"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for generation"
    )
    args = parser.parse_args()

    model, tokenizer = load_fine_tuned_model(
        args.peft_model_path, base_model_name=models[args.model_type]
    )

    test_dataset = load_dataset("openai/gsm8k", "main", split="test")
    results = evaluate_model(
        model,
        tokenizer,
        test_dataset,
        batch_size=args.batch_size,
        gen_max_length=args.gen_max_length,
    )
    print("Evaluation Results:")
    print(f"Correct: {results['num_correct']}, Total: {results['num_total']}")
    print(f"Accuracy: {results['accuracy']:.2%}")

    results_path = os.path.join(args.peft_model_path, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
