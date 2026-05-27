#!/bin/bash
#SBATCH --job-name=logzpo
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

cd /nfs/roberts/project/pi_tkf6/as4643/projects/openpi

# activate your project environment
source .venv/bin/activate

# run
python -u -m scripts.logzpo.eval_auroc
