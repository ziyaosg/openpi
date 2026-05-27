#!/bin/bash
#SBATCH --job-name=logzpo
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --requeue

set -euo pipefail
cd /nfs/roberts/project/pi_tkf6/as4643/projects/openpi
source .venv/bin/activate
python -u -m scripts.logzpo.train_logzpo