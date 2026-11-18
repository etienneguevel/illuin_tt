#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=4
#SBATCH --error=/home/guevel/logs/%j_0_log.err
#SBATCH --gpus-per-node=1
#SBATCH --job-name=embedding
#SBATCH --nodes=1
#SBATCH --mem=38000
#SBATCH --output=/home/guevel/logs/%j_0_log.out
#SBATCH --partition=hard
#SBATCH --nodelist=lizzy
#SBATCH --time=1500

cd /home/guevel/projects/illuin_tt
conda deactivate
source .venv/bin/activate

python illuin_tt/scripts/embedding_method.py --config-path configs/Qwen3-Embedding-4B_code.yaml
