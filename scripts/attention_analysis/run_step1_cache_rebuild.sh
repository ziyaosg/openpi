#!/bin/bash
#SBATCH --job-name=rebuild-caches
#SBATCH --output=/home/zs377/logs/rebuild_step1_%j.out
#SBATCH --error=/home/zs377/logs/rebuild_step1_%j.err
#SBATCH --time=5:30:00
#SBATCH --partition=devel
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

source /etc/profile.d/modules.sh
module load miniconda
conda activate openpi

cd /home/zs377/project_pi_tkf6/zs377/projects/openpi_liberoplus

echo "[$(date)] Deleting old caches..."
rm -f /nfs/roberts/scratch/pi_tkf6/zs377/analysis_conformal_v2/feature_cache/*.pkl

echo "[$(date)] conformal_analysis_v2.py — rebuilding 3 group caches (330 features)..."
python scripts/attention_analysis/conformal_analysis_v2.py 2>&1

echo "[$(date)] Step 1 done. Submitting step 2..."
sbatch /home/zs377/project_pi_tkf6/zs377/projects/openpi_liberoplus/scripts/attention_analysis/run_step2_analysis.sh
