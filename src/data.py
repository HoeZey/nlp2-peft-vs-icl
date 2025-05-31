from collections import Counter
import re
import torch
from torch import IntTensor


def remove_left_right_cmds(s: str) -> str:
    return s.replace("\\left", "").replace("\\right", "").replace(" ", "")


def last_boxed_only_string(response: str):
    idx = response.rfind("\\boxed")
    if idx < 0:
        idx = response.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(response):
        if response[i] == "{":
            num_left_braces_open += 1
        if response[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return None if right_brace_idx is None else response[idx + 7 : right_brace_idx]


def parse_answer(g: str) -> int:
    answer = re.findall(r"\d+", g)
    try:
        return min(int(answer[-1]), 999_999_999_999)
    except:
        pass
    return -1


def majority_vote(answers: list[int]) -> int:
    if len(answers) == 0:
        return answers[0]
    return Counter(answers).most_common(1)[0][0]


def get_answer_from_outputs(generated: list[list[str]]) -> torch.IntTensor:
    answers = [majority_vote([parse_answer(g) for g in gen]) for gen in generated]
    return torch.tensor(answers, dtype=int)


class Processor:
    def get_solutions(self, questions) -> IntTensor:
        pass

    def get_questions(self, questions) -> list[dict[str, str]]:
        pass

    def get_questions_from_icl_datset(self, dataset) -> list[dict[str, str]]:
        pass

    def get_answer_from_outputs(self, generated) -> IntTensor:
        pass

    def process_examples(self, questions) -> list[str]:
        pass

    def get_correct(self, answers, solutions) -> int:
        pass


class GSM8KProcessor:
    """Processor for the GSM8K dataset. Based on openai/gsm8k."""

    def get_solutions(self, questions) -> IntTensor:
        return torch.tensor([parse_answer(q) for q in questions["answer"]], dtype=int)

    def get_questions(self, questions) -> list[str]:
        return questions["question"]

    def get_questions_from_icl_datset(self, dataset) -> list[dict[str, str]]:
        return [d["question"] for d in dataset]

    def get_answer_from_outputs(self, generated) -> IntTensor:
        return get_answer_from_outputs(generated)

    def process_examples(self, examples) -> str:
        prompt = ""
        for example in examples:
            # prompt += f"Example {batch + 1}\n"
            prompt += f"Question: {example['question']}\n"
            split = example["answer"].split("####")
            prompt += (
                f"Solution: {split[0].strip()}. The answer is {split[1].strip()}\n\n"
            )
        return prompt

    def get_correct(self, answers, solutions) -> int:
        return (answers == solutions).sum().item()


class MATHProcessor:
    """Processor for the MATH dataset. Based on nlile/hendrycks-MATH-benchmark."""

    def get_solutions(self, questions) -> list[str]:
        return [remove_left_right_cmds(q) for q in questions["answer"]]

    def get_questions(self, questions) -> list[str]:
        return questions["problem"]

    def get_questions_from_icl_datset(self, dataset) -> list[dict[str, str]]:
        return [d["problem"] for d in dataset]

    def get_answer_from_outputs(self, generated) -> list[str]:
        return [
            majority_vote([last_boxed_only_string(g) for g in gen]) for gen in generated
        ]

    def process_examples(self, examples) -> str:
        prompt = ""
        for example in examples:
            # prompt += f"Example {batch + 1}\n"
            prompt += f"Question: {example['problem']}\n"
            prompt += f"Solution: {example['solution']}\n\n"
            prompt += "#" * 100 + "\n\n"

        return prompt

    def get_correct(self, answers, solutions) -> int:
        return sum(
            a == s
            for a, s in zip(answers, solutions)
            if a is not None and s is not None
        )


class AIMEProcessor:
    """
    AIME processor for the AIME dataset.
    Based on rawsh/aime_2025 for test set and Maxwell-Jia/AIME_2024 for train set.
    """

    def get_solutions(self, questions) -> IntTensor:
        return torch.tensor([int(q) for q in questions["answer"]], dtype=int)

    def get_questions(self, questions) -> list[str]:
        return questions["problem"]

    def get_questions_from_icl_datset(self, dataset) -> list[dict[str, str]]:
        return [d["Problem"] for d in dataset]

    def get_answer_from_outputs(self, generated) -> IntTensor:
        return get_answer_from_outputs(generated)

    def process_examples(self, examples) -> str:
        prompt = ""
        for example in examples:
            # prompt += f"Example {batch + 1}\n"
            prompt += f"Question: {example['Problem']}\n"
            prompt += f"Solution: {example['Solution']}. The answer is {example['Answer']}\n\n"
        return prompt

    def get_correct(self, answers, solutions) -> int:
        return (answers == solutions).sum().item()
