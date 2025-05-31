import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig
from datasets import load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from dotenv import load_dotenv
import os
import re
import argparse
import wandb
import math

from evaluate import evaluate_model


models = {
    "llama2": "meta-llama/Llama-2-7b-hf",
    "llama3": "meta-llama/Llama-3.1-8B",
}

sweep_config = {
    "name": "lora-sweep",
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "val_accuracy"},
    "parameters": {
        "learning_rate": {
            "distribution": "log_normal",
            "mu": math.log(1e-4),
            "sigma": math.log(10) ** 0.5,
        },
        "weight_decay": {
            "distribution": "log_normal",
            "mu": math.log(1e-3),
            "sigma": math.log(10) ** 0.5,
        },
        "gradient_accumulation_steps": {
            "values": [1, 2, 4],
        },
        "lora_alpha": {
            "values": [16, 32, 64],
        },
        "r": {
            "values": [8, 16, 32],
        },
        "lora_dropout": {
            "distribution": "log_normal",
            "mu": math.log(0.1),
            "sigma": math.log(2) ** 0.5,
        }
    },
}


def rm_calculator_instructions(example):
    example['answer'] = re.sub(r'<<.*?>>', '', example['answer'])
    return example


def formatting_func(example):
    if not isinstance(example['question'], list):
        return [f"### Question: {example['question']}\n### Answer: {example['answer']}"]

    output_texts = []
    for i in range(len(example['question'])):
        text = f"### Question: {example['question'][i]}\n### Answer: {example['answer'][i]}"
        output_texts.append(text)
    return output_texts


def train(model_type, config={}):
    if model_type not in models:
        raise ValueError(f"Model {model_type} not found. Available models: {list(models.keys())}")

    model_name = models[model_type]
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=os.getenv("HUGGINGFACE_TOKEN"),
    )
    # tokenizer.add_special_tokens({'pad_token': '<PAD>'})
    tokenizer.pad_token_id = tokenizer.unk_token_id

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=os.getenv("HUGGINGFACE_TOKEN"),
        quantization_config=bnb_config,
        device_map='auto',
        # pad_token_id=tokenizer.pad_token_id,
    )
    print("device:", model.device)

    lora_config = dict(
        r=32,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    for key in lora_config.keys():
        lora_config[key] = config.pop(key, lora_config[key])
    lora_config = LoraConfig(
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model.add_adapter(lora_config)
    print(model)

    dataset = load_dataset("openai/gsm8k", "main")
    dataset = dataset.map(rm_calculator_instructions)
    train_dataset, val_dataset = dataset["train"].train_test_split(test_size=0.01).values()
    test_dataset = dataset["test"]

    response_template_with_context = "\n### Answer:"
    response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[2:]
    collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

    train_config = dict(
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        optim="paged_adamw_32bit",
        bf16=True,
        eval_strategy="steps",
        logging_steps=10,
        eval_steps=20,
        save_steps=50,
        learning_rate=1e-4,
        # max_grad_norm=0.3,
        # max_steps=500,
        num_train_epochs=1,
        warmup_steps=10,
        weight_decay=0.01,
        lr_scheduler_type="constant",
        gradient_checkpointing=True,
        push_to_hub=False,
    ) | config
    training_arguments = TrainingArguments(
        run_name=wandb.run.name,
        report_to="wandb",
        output_dir=f"./output/{wandb.run.name}",
        **train_config,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        # packing=True,
        # max_seq_length=1024,
        processing_class=tokenizer,
        formatting_func=formatting_func,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(f"output/{wandb.run.name}")
    
    eval_results = evaluate_model(
        model,
        tokenizer,
        val_dataset,
        batch_size=16,
    )
    print("Validation accuracy:", eval_results["accuracy"])
    wandb.log("val_accuracy", eval_results["accuracy"])


def train_sweep(model_type):
    train(model_type, config=wandb.config)
    
if __name__ == "__main__":
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Train a model with LoRA.")
    parser.add_argument(
        "--model_type",
        type=str,
        choices=models.keys(),
        required=True,
        help="Type of model to train (e.g., llama2, llama3)."
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run a hyperparameter sweep."
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=1,
        help="Number of runs for the sweep."
    )
    
    args = parser.parse_args()
    if args.sweep:
        sweep_id = wandb.sweep(
            sweep=sweep_config, project="nlp2", entity="leonardhorns"
        )
        wandb.agent(
            sweep_id,
            function=lambda: train_sweep(args.model_type),
            count=args.num_runs,
        )
    else:
        wandb.init(
            project="nlp2",
            entity="leonardhorns",
        )
        train(args.model_type)
    wandb.finish()
