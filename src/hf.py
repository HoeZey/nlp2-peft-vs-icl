import os
from dotenv import load_dotenv
from huggingface_hub import login


def hf_login():
    load_dotenv("./secrets.env")
    login(token=os.environ.get("HF_ACCESS_TOKEN"))
