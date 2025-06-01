# NLP2 Project
The project explores the effectiveness of in-context learning vs. finetuning for math benchmarks.

## Setup
The repository uses uv repository manager. Install uv via pip then run
- `sh script/create_venv.sh`
- `sbatch script/create_conda_venv.sh`

Create a `secrets.env` file with your huggingface access token like:
`HF_ACCESS_TOKEN=<token>`

## Running the evaluation and training jobs
You can simply run the jobs by running the `sh scripts/run_*.sh` files or sbatch the jobs under `jobs/lora/train/`.