import os
import wandb
import argparse
from hf import hf_login
from lora.train import train, train_sweep
from lora.utils import models


def main():
    hf_login()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser(description="Train a model with LoRA.")
    parser.add_argument(
        "--model_type",
        type=str,
        choices=models.keys(),
        required=True,
        help="Type of model to train (e.g., llama2, llama3).",
    )
    parser.add_argument(
        "--sweep", action="store_true", help="Run a hyperparameter sweep."
    )
    parser.add_argument(
        "--sweep_id",
        help="ID of the sweep to pass to the agent",
    )
    parser.add_argument(
        "--num_runs", type=int, default=1, help="Number of runs for the sweep."
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to a yaml config with lora and training arguments (only if --sweep is not set)",
    )

    args = parser.parse_args()
    if args.sweep:
        wandb.agent(
            args.sweep_id,
            function=lambda: train_sweep(args.model_type),
            count=args.num_runs,
            entity="leonardhorns",
            project="nlp2",
        )
    else:
        if args.config is not None:
            import yaml

            with open(args.config, "r") as f:
                config = yaml.safe_load(f)

        wandb.init(
            project="nlp2",
            entity="leonardhorns",
        )
        train(args.model_type, config=config)


if __name__ == "__main__":
    main()
