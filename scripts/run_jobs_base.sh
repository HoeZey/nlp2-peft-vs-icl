#!/bin/bash          
WAIT=5

# Llama3.1-Base
sbatch jobs/llama3_base/gsm8k/run_eval_llama3_0s.job
sleep $WAIT
sbatch jobs/llama3_base/gsm8k/random/run_eval_llama3_1s.job
sleep $WAIT
# sbatch jobs/llama3_base/gsm8k/random/run_eval_llama3_2s.job
# sleep $WAIT
sbatch jobs/llama3_base/gsm8k/random/run_eval_llama3_4s.job
sleep $WAIT
sbatch jobs/llama3_base/gsm8k/sim/run_eval_llama3_1s.job
sleep $WAIT
# sbatch jobs/llama3_base/gsm8k/sim/run_eval_llama3_2s.job
# sleep $WAIT
sbatch jobs/llama3_base/gsm8k/sim/run_eval_llama3_4s.job
sleep $WAIT

sbatch jobs/llama3_base/math/run_eval_llama3_0s.job
sleep $WAIT
sbatch jobs/llama3_base/math/random/run_eval_llama3_1s.job
sleep $WAIT
# sbatch jobs/llama3_base/math/random/run_eval_llama3_2s.job
# sleep $WAIT
sbatch jobs/llama3_base/math/random/run_eval_llama3_4s.job
sleep $WAIT
sbatch jobs/llama3_base/math/sim/run_eval_llama3_1s.job
sleep $WAIT
# sbatch jobs/llama3_base/math/sim/run_eval_llama3_2s.job
# sleep $WAIT
sbatch jobs/llama3_base/math/sim/run_eval_llama3_4s.job