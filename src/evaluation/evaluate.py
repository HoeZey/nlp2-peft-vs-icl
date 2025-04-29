import json
from pathlib import Path
import re
import torch
from torch.utils.data import DataLoader, RandomSampler
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from datasets import Dataset


class FewshotSampler:
    def __init__(self, dataset, k, seed=None):
        self.dataset = dataset
        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(seed)
        self.sampler = RandomSampler(self.dataset, num_samples=k, generator=generator)

    def sample(self):
        return [self.dataset[i] for i in self.sampler]


def extract_answer(g: str):
    answer = re.findall(r"\d+", g)
    return (
        min(int(answer[-1]), 999_999_999_999)
        if answer is not None and len(answer) > 0
        else -1
    )


def extract_answers(generated: list[str]) -> torch.IntTensor:
    return torch.tensor([extract_answer(g) for g in generated], dtype=int)


def evaluate(
    model: LLM,
    sampling_params: SamplingParams,
    eval_dataset: Dataset,
    batch_size: int,
    answers_file: Path,
    output_file: Path,
    fewshot_sampler: RandomSampler = None,
    lora_request: LoRARequest = None,
    system_prompt: str = "",
):
    correct = 0
    model_answers: dict[int, int] = (
        json.load(answers_file.open("r")) if answers_file.stat().st_size > 0 else dict()
    )
    model_output: dict[int, int] = (
        json.load(output_file.open("r")) if output_file.stat().st_size > 0 else dict()
    )

    for batch, questions in enumerate(
        DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    ):
        if str((batch + 1) * batch_size - 1) in model_answers:
            pass
        print(f"Processing batch {batch}")

        solutions = extract_answers(questions["answer"])
        prompts = []
        for q in questions["question"]:
            prompt = f"{system_prompt}\n\n"
            if fewshot_sampler:
                for batch, example in enumerate(fewshot_sampler.sample()):
                    # prompt += f"Example {batch + 1}\n"
                    prompt += f"Q: {example['question']}\n"
                    split = example["answer"].split(" #### ")
                    prompt += f"A: {split[0]} The answer is {split[1]}\n\n"
            prompt += f"Q: {q}\nA: "
            prompts.append(prompt)

        outputs = model.generate(
            prompts,
            sampling_params,
            lora_request=lora_request,
        )
        generated = [o.outputs[0].text for o in outputs]
        answers = extract_answers(generated)
        correct += (answers == solutions).sum()

        for i, g in enumerate(generated):
            model_output[str(batch * batch_size + i)] = g

        print(f"Solutions: {solutions.tolist()}")
        print(f"Answers: {answers.tolist()}")

        for i, (a, g) in enumerate(zip(answers.tolist(), generated)):
            model_answers[str(batch * batch_size + i)] = a
            model_output[str(batch * batch_size + i)] = g

        json.dump(model_answers, answers_file.open("w"), indent=4)
        json.dump(model_output, output_file.open("w"), indent=4)

    print(f"Accuracy: {100 * correct / len(eval_dataset):.2f}")
