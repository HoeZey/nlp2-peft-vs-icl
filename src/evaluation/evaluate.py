from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


evaluate
llm = LLM(model="meta-llama/Llama-2-7b-hf", enable_lora=True)
