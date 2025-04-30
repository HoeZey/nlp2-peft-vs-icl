import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig
from datasets import load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from dotenv import load_dotenv
import os
import re

load_dotenv()

model_name = "meta-llama/Llama-2-7b-hf"

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

lora_config = LoraConfig(
    r=32,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)

model.add_adapter(lora_config)
print(model)


def rm_calculator_instructions(example):
    example['answer'] = re.sub(r'<<.*?>>', '', example['answer'])
    return example


dataset = load_dataset("openai/gsm8k", "main")
dataset = dataset.map(rm_calculator_instructions)
train_dataset, val_dataset = dataset["train"].train_test_split(test_size=0.01).values()
test_dataset = dataset["test"]


def formatting_func(example):
    if not isinstance(example['question'], list):
        return [f"### Question: {example['question']}\n### Answer: {example['answer']}"]

    output_texts = []
    for i in range(len(example['question'])):
        text = f"### Question: {example['question'][i]}\n### Answer: {example['answer'][i]}"
        output_texts.append(text)
    return output_texts


response_template_with_context = "\n### Answer:"
response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[2:]
collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

training_arguments = TrainingArguments(
    run_name="snellius-test-run",
    output_dir="./output/llama2-extended",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    optim="paged_adamw_32bit",
    bf16=True,
    eval_strategy="steps",
    logging_steps=10,
    eval_steps=20,
    save_steps=20,
    learning_rate=1e-4,
    # max_grad_norm=0.3,
    max_steps=500,
    warmup_steps=10,
    # warmup_ratio=0.03,
    lr_scheduler_type="constant",
    gradient_checkpointing=True,
    push_to_hub=False,
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

num_examples = 5
for i in range(num_examples):
    example = dataset["test"][i]
    print(f"============ Example {i + 1}: ============")
    print(example)
    prompt = f"### Question: {example['question']}\n### Answer:"
    output = model.generate(
        **tokenizer(prompt, return_tensors="pt").to(model.device),
        do_sample=False
    )
    print("Model answer:", tokenizer.decode(output[0]), sep="\n")
