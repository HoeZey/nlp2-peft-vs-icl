import re
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorWithPadding
from datasets import load_dataset
from dotenv import load_dotenv
import os
from argparse import ArgumentParser
from tqdm import tqdm

load_dotenv()


ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"


def extract_answer(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def is_correct(model_completion, gt_answer):
    gt_answer = extract_answer(gt_answer)
    assert gt_answer != INVALID_ANS
    return extract_answer(model_completion) == gt_answer


def rm_calculator_instructions(example):
    example['answer'] = re.sub(r'<<.*?>>', '', example['answer'])
    return example


def formatting_func(example):
    if not isinstance(example['question'], list):
        return [f"### Question: {example['question']}\n### Answer:"]

    output_texts = []
    for i in range(len(example['question'])):
        text = f"### Question: {example['question'][i]}\n### Answer:"
        output_texts.append(text)
    return output_texts


class CustomDataCollator(DataCollatorWithPadding):
    def __init__(self, preserve_fields_keys=None, *args, **kwargs):
        self.preserve_fields_keys = preserve_fields_keys or []
        super().__init__(*args, **kwargs)

    def __call__(self, features):
        preserved_fields = {}
        for key in self.preserve_fields_keys:
            preserved_fields[key] = [f.pop(key) for f in features]

        batch = super().__call__(features)

        batch |= preserved_fields
        return batch


def load_fine_tuned_model(peft_model_path, base_model_name="meta-llama/Llama-2-7b-hf"):
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        token=os.getenv("HUGGINGFACE_TOKEN"),
    )
    tokenizer.pad_token_id = tokenizer.unk_token_id
    tokenizer.padding_side = "left"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        token=os.getenv("HUGGINGFACE_TOKEN"),
        quantization_config=bnb_config,
        device_map='auto',
    )
    
    model = PeftModel.from_pretrained(
        base_model,
        peft_model_path,
        device_map='auto',
    )
    model = model.merge_and_unload()
    return model, tokenizer


def prepare_dataloader(tokenizer, dataset, batch_size=16):
    dataset = dataset.map(rm_calculator_instructions)
    dataset = dataset.map(lambda example: tokenizer(formatting_func(example)), batched=True)
    dataset = dataset.select_columns(["input_ids", "attention_mask", "answer"])

    collator = CustomDataCollator(
        tokenizer=tokenizer,
        preserve_fields_keys=["answer"],
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=4,
    )
    return dataloader


def evaluate_model(model, tokenizer, dataset, batch_size=16):
    device = model.device
    data_loader = prepare_dataloader(tokenizer, dataset, batch_size=batch_size)

    correct_count = 0
    for batch in tqdm(data_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        answers = batch["answer"]

        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=args.gen_max_length,
                do_sample=False,
            ).cpu()
    
        completions = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for completion, answer in zip(completions, answers):
            if is_correct(completion, answer):
                correct_count += 1
    return {
        "num_correct": correct_count,
        "num_total": len(data_loader.dataset),
        "accuracy": correct_count / len(data_loader.dataset),
    }


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--peft_model_path", type=str, required=True, help="Path to the fine-tuned model")
    parser.add_argument("--gen_max_length", type=int, default=512, help="Max length for generation")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for generation")
    args = parser.parse_args()

    model, tokenizer = load_fine_tuned_model(args.peft_model_path)
    device = model.device

    test_dataset = load_dataset("openai/gsm8k", "main", split="test")
    results = evaluate_model(model, tokenizer, test_dataset, batch_size=args.batch_size)

    print(f"Correct: {results["num_correct"]}, Total: {results["num_total"]}")
    print(f"Accuracy: {results["accuracy"]:.2%}")
