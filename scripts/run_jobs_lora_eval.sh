#!/bin/bash          
WAIT=10  # wait between jobs because scheduling jobs too fast causes errors

# Llama3.1-Base
sbatch jobs/lora/gsm8k/run_eval_llama3_0s.job
sleep $WAIT
sbatch jobs/lora/math/run_eval_llama3_0s.job
