# From https://github.com/huggingface/trl/blob/main/examples/research_projects/stack_llama/scripts/supervised_finetuning.py

import os
import argparse
from transformers import AutoModelForCausalLM, TrainingArguments, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer

from training.peft import create_datasets


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument(
        "--dataset_name", type=str, default="lvwerra/stack-exchange-paired"
    )
    parser.add_argument("--subset", type=str, default="data/finetune")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--size_valid_set", type=int, default=4000)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--shuffle_buffer", type=int, default=5000)

    parser.add_argument("--seq_length", type=int, default=1024)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--eos_token_id", type=int, default=49152)

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--num_warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.05)

    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_freq", default=1, type=int)
    parser.add_argument("--eval_freq", default=1000, type=int)
    parser.add_argument("--save_freq", default=1000, type=int)

    return parser.parse_args()


def main():
    args = get_args()

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        dataloader_drop_last=True,
        eval_strategy="steps",
        max_steps=args.max_steps,
        eval_steps=args.eval_freq,
        save_steps=args.save_freq,
        logging_steps=args.log_freq,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.num_warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        fp16=args.fp16,
        bf16=args.bf16,
        weight_decay=args.weight_decay,
        run_name="llama-7b-finetuned",
        report_to="wandb",
        ddp_find_unused_parameters=False,
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    train_data, eval_data = create_datasets(tokenizer, args)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        load_in_8bit=True,
        # device_map={"": Accelerator().process_index},
    )

    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        peft_config=lora_config,
        packing=True,
    )

    trainer.train()

    trainer.model.save_pretrained(os.path.join(args.output_dir, "final_checkpoint/"))


if __name__ == "__main__":
    main()
