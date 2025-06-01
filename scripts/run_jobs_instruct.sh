#!/bin/bash          
WAIT=10  # wait between jobs because scheduling jobs too fast causes errors

# Llama3.1-Instruct
sbatch jobs/llama3_instruct/gsm8k/run_eval_llama3_0s.job
sleep $WAIT
sbatch jobs/llama3_instruct/gsm8k/random/run_eval_llama3_1s.job
sleep $WAIT
sbatch jobs/llama3_instruct/gsm8k/random/run_eval_llama3_4s.job
sleep $WAIT
sbatch jobs/llama3_instruct/gsm8k/sim/run_eval_llama3_1s.job
sleep $WAIT
sbatch jobs/llama3_instruct/gsm8k/sim/run_eval_llama3_4s.job
sleep $WAIT

sbatch jobs/llama3_instruct/math/run_eval_llama3_0s.job
sleep $WAIT
sbatch jobs/llama3_instruct/math/random/run_eval_llama3_1s.job
sleep $WAIT
sbatch jobs/llama3_instruct/math/random/run_eval_llama3_4s.job
sleep $WAIT
sbatch jobs/llama3_instruct/math/sim/run_eval_llama3_1s.job
sleep $WAIT
sbatch jobs/llama3_instruct/math/sim/run_eval_llama3_4s.job