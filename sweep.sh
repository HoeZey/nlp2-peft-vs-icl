#!/bin/bash

set -e

# Default number of runs
NUM_RUNS=5

# Check if a custom number of runs was passed as an argument
if [ ! -z "$1" ]; then
  NUM_RUNS=$1
fi

# Create sweep and store its ID
#SWEEP_ID=$(python -c "import wandb; from train import sweep_config; print(wandb.sweep(sweep=sweep_config, project='nlp2', entity='leonardhorns'))")
#SWEEP_ID=$(wandb sweep sweep_config.yaml)
SWEEP_ID=$(wandb sweep --project nlp2 sweep_config.yaml 2>&1 | awk '/Creating sweep with ID:/ { print $NF }')

echo "Sweep created with ID: $SWEEP_ID"

# Launch runs via Accelerate
for i in $(seq 1 $NUM_RUNS); do
  accelerate launch train.py --model_type llama3 --sweep --sweep_id $SWEEP_ID
done
