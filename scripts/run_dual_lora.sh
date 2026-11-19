#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=4
#SBATCH --error=/home/guevel/logs/%j_0_log.err
#SBATCH --gpus-per-node=1
#SBATCH --job-name=dual-lora-embedding
#SBATCH --nodes=1
#SBATCH --mem=38000
#SBATCH --output=/home/guevel/logs/%j_0_log.out
#SBATCH --partition=hard
#SBATCH --nodelist=lizzy
#SBATCH --time=1500

cd /home/guevel/projects/illuin_tt
source .venv/bin/activate

uv run python -m illuin_tt.scripts.dual_lora_finetune --config-path configs/Qwen3-Embedding-4B_dual_lora.yaml
