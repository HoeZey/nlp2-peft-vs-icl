import torch
from torch.utils.data import DataLoader, RandomSampler
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from datasets import Dataset


def add_examples_to_prompt(prompt: str, sampler: RandomSampler) -> str:
    for question, answer in next(sampler):
        prompt += question + str(answer)
    return prompt


def extract_answer(text: list[str]) -> torch.IntTensor:
    return 0


def evaluate(
    model: LLM,
    sampling_params: SamplingParams,
    lora_request: LoRARequest,
    eval_dataset: Dataset,
    batch_size: int,
    k=0,
    icl_dataset: Dataset = None,
):
    correct = 0
    if k > 0:
        sampler = RandomSampler(icl_dataset, num_samples=k)

    for questions, solutions in DataLoader(eval_dataset):
        if k > 0:
            questions = [add_examples_to_prompt(q, sampler) for q in questions]

        outputs = model.generate(
            questions,
            sampling_params,
            lora_request=lora_request,
        )
        answer = extract_answer(outputs[0].outputs[0].text)

        correct += answer == solutions

    print(f"Accuracy: {100 * correct / len(eval_dataset):.2f}")
