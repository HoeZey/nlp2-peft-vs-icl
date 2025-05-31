import os
from dotenv import load_dotenv
from huggingface_hub import login


def hf_login(path: str = "./secrets.env"):
    load_dotenv(path)
    login(token=os.environ.get("HF_ACCESS_TOKEN"))
