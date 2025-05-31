from dataclasses import dataclass
import os
from datasets import load_dataset
from dotenv import load_dotenv


def get_dataset(name: str, split=None, env_path="./secrets.env"):
    load_dotenv(env_path)
    return load_dataset(name, split=split, token=os.getenv("HF_ACCESS_TOKEN"))


def get_aime_1983_to_2024(split=None, env_path="./secrets.env"):
    return get_dataset("gneubig/aime-1983-2024", split, env_path)


def get_aime_2025(split=None, env_path="./secrets.env"):
    return get_dataset("opencompass/AIME2025", split, env_path)


@dataclass(kw_only=True)
class DataLoaderParams:
    batch_size: int
    shuffle: bool
    num_workers: int = 0
