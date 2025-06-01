#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=CONDA
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=0:30:00
#SBATCH --output=outputs/venv/%A.out

module purge
module load 2024
module load Anaconda3/2024.06-1

cd $HOME/nlp2/

uv pip compile -q pyproject.toml -o requirements.txt
conda env remove --name nlp2 -y
conda env create -f environment.yaml