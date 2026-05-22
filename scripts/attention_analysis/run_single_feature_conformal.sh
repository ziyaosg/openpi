#!/bin/bash
#SBATCH --job-name=conformal-single
#SBATCH --output=/home/zs377/logs/conformal_single_%j.out
#SBATCH --error=/home/zs377/logs/conformal_single_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

source /etc/profile.d/modules.sh
module load miniconda
conda activate openpi

cd /home/zs377/project_pi_tkf6/zs377/projects/openpi_liberoplus
python scripts/attention_analysis/single_feature_conformal.py
