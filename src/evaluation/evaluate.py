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


def extract_answer(generated: list[str]) -> torch.IntTensor:
    return torch.tensor([int(g.split("####")[-1]) for g in generated], dtype=int)


def evaluate(
    model: LLM,
    sampling_params: SamplingParams,
    eval_dataset: Dataset,
    batch_size: int,
    fewshot_sampler: RandomSampler = None,
    lora_request: LoRARequest = None,
    system_prompt: str = "",
):
    correct = 0

    for questions in DataLoader(eval_dataset, batch_size=batch_size, shuffle=False):
        solutions = extract_answer([q["answer"] for q in questions])
        prompts = []
        for q in questions:
            prompt = system_prompt
            if fewshot_sampler:
                for i, example in enumerate(fewshot_sampler.sample()):
                    prompt += f"Example {i + 1}\nQuestion:\n"
                    prompt += example["question"] + "\nAnswer:\n"
                    prompt += example["answer"] + "\n\n"
            prompt += "Test:\nQuestion:\n" + q["question"]
            prompts.append(prompt)

        outputs = model.generate(
            questions,
            sampling_params,
            lora_request=lora_request,
        )
        answers = extract_answer(outputs[0].outputs[0].text)

        correct += (answers == solutions).sum()

    print(f"Accuracy: {100 * correct / len(eval_dataset):.2f}")
