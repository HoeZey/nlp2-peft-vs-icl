import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader, RandomSampler
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from data import (
    MATHProcessor,
    Processor,
    last_boxed_only_string,
    remove_left_right_cmds,
)


class FewshotSampler:
    def sample(self, x=None):
        pass


class RandomFewshotSampler:
    def __init__(self, dataset, k, seed=None):
        self.dataset = dataset
        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(seed)
        self.sampler = RandomSampler(self.dataset, num_samples=k, generator=generator)

    def sample(self, x=None):
        return [self.dataset[i] for i in self.sampler]


class SimilarityFewshotSampler:
    def __init__(self, dataset, processor, k, model, seed=None):
        self.dataset = dataset
        self.k = k
        self.seed = seed
        self.model = SentenceTransformer(model)
        self.embeddings = self.model.encode(
            processor.get_questions_from_icl_datset(dataset)
        )

    def sample(self, x):
        similarities = self.model.similarity(self.model.encode(x), self.embeddings)
        indices_top_k = similarities.flatten().argsort().tolist()[-self.k :][::-1]
        return [self.dataset[i] for i in indices_top_k]


def get_prompts(questions, processor, system_prompt, fewshot_sampler):
    prompts = []
    for q in processor.get_questions(questions):
        prompt = f"{system_prompt}\n\n"
        if fewshot_sampler is not None:
            prompt += processor.process_examples(fewshot_sampler.sample(q))
        prompt += f"Question: {q}\nSolution: "
        prompts.append(prompt)
    return prompts


def get_messages(questions, processor, system_prompt, fewshot_sampler):
    prompts = []
    for q in processor.get_questions(questions):
        prompt = [{"role": "system", "content": f"{system_prompt}"}]
        question = ""
        if fewshot_sampler is not None:
            question += processor.process_examples(fewshot_sampler.sample(q))
        question += f"Question: {q}\nSolution: "
        prompt.append({"role": "user", "content": question})
        prompts.append(prompt)
    return prompts


def log_outputs(
    generated,
    answers,
    batch,
    batch_size,
    answers_file,
    output_file,
    model_answers,
    model_output,
):
    for i, g in enumerate(generated):
        model_output[str(batch * batch_size + i)] = g

    for i, (a, g) in enumerate(zip(answers, generated)):
        model_answers[str(batch * batch_size + i)] = a
        model_output[str(batch * batch_size + i)] = g

    print(f"Writing batch {batch} to files")
    json.dump(model_answers, answers_file.open("w"), indent=4)
    json.dump(model_output, output_file.open("w"), indent=4)


def evaluate(
    model: LLM,
    sampling_params: SamplingParams,
    eval_dataset: Dataset,
    processor: Processor,
    batch_size: int,
    answers_file: Path,
    output_file: Path,
    fewshot_sampler: FewshotSampler = None,
    lora_request: LoRARequest = None,
    system_prompt: str = "",
):
    correct = 0
    dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    model_answers: dict[int, int] = (
        json.load(answers_file.open("r")) if answers_file.stat().st_size > 0 else dict()
    )
    model_output: dict[int, int] = (
        json.load(output_file.open("r")) if output_file.stat().st_size > 0 else dict()
    )

    for batch, questions in enumerate(dataloader):
        if str((batch + 1) * batch_size - 1) in model_answers:
            pass
        print(f"Processing batch {batch}")

        solutions = processor.get_solutions(questions)
        is_chat_model = "instruct" in answers_file.name.lower()
        if is_chat_model:
            outputs = model.chat(
                get_messages(questions, processor, system_prompt, fewshot_sampler),
                sampling_params,
                lora_request=lora_request,
            )
        else:
            outputs = model.generate(
                get_prompts(questions, processor, system_prompt, fewshot_sampler),
                sampling_params,
                lora_request=lora_request,
            )
        generated = [[o.text for o in out.outputs] for out in outputs]
        answers = processor.get_answer_from_outputs(generated)
        correct += processor.get_correct(answers, solutions)

        if type(processor) is MATHProcessor:
            print(f"Solutions: {solutions}")
            print(f"Answers: {answers}")
        else:
            print(f"Solutions: {solutions.tolist()}")
            print(f"Answers: {answers.tolist()}")

        log_outputs(
            generated,
            answers.tolist() if not isinstance(answers, list) else answers,
            batch,
            batch_size,
            answers_file,
            output_file,
            model_answers,
            model_output,
        )

    print(f"Accuracy: {100 * correct / len(eval_dataset):.2f}")


def check_answers(eval_dataset: Dataset, processor: Processor, answers_file: Path):
    answers_dict = json.load(answers_file.open("r"))
    answers = list(
        remove_left_right_cmds(answers_dict[str(i)]) for i in range(len(eval_dataset))
    )
    solutions = processor.get_solutions(eval_dataset)
    answers_print = answers
    if type(processor) is MATHProcessor:
        print(f"Solutions: {solutions}")
        answers = [
            (
                remove_left_right_cmds(last_boxed_only_string(a))
                if a is not None
                else None
            )
            for a in answers
        ]
        answers_print = answers
    else:
        print(f"Solutions: {solutions.tolist()}")
        answers = torch.tensor(answers, dtype=torch.int32)
    print(f"Answers: {answers_print}")

    correct = processor.get_correct(answers, solutions)
    print(f"Accuracy: {100 * correct / len(eval_dataset):.2f}")
