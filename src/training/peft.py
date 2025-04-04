# From https://github.com/huggingface/trl/blob/main/examples/research_projects/stack_llama/scripts/supervised_finetuning.py

from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer
from trl.trainer import ConstantLengthDataset


def chars_token_ratio(dataset, tokenizer, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = prepare_sample_text(example)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_characters / total_tokens


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def prepare_sample_text(example):
    """Prepare the text from a sample of the dataset."""
    text = f"Question: {example['question']}\n\nAnswer: {example['response_j']}"
    return text


def create_datasets(tokenizer, args):
    dataset = load_dataset(
        args.dataset_name,
        data_dir=args.subset,
        split=args.split,
        use_auth_token=True,
        num_proc=args.num_workers if not args.streaming else None,
        streaming=args.streaming,
    )
    if args.streaming:
        print("Loading the dataset in streaming mode")
        valid_data = dataset.take(args.size_valid_set)
        train_data = dataset.skip(args.size_valid_set)
        train_data = train_data.shuffle(buffer_size=args.shuffle_buffer, seed=args.seed)
    else:
        dataset = dataset.train_test_split(test_size=0.005, seed=args.seed)
        train_data = dataset["train"]
        valid_data = dataset["test"]
        print(
            f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}"
        )

    chars_per_token = chars_token_ratio(train_data, tokenizer)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        formatting_func=prepare_sample_text,
        infinite=True,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        formatting_func=prepare_sample_text,
        infinite=False,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
    )
    return train_dataset, valid_dataset


def main():
    args = get_args()

    train_data = load_dataset(
        args.dataset_name,
        data_dir=args.subset,
        split=args.split,
        use_auth_token=True,
        num_proc=args.num_workers if not args.streaming else None,
        streaming=args.streaming,
    )

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
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        load_in_8bit=True,
        # device_map={"": Accelerator().process_index},
    )

    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=train_data,
        peft_config=lora_config,
        packing=True,
    )

    trainer.train()
