#!/bin/bash
#SBATCH --job-name=rebuild-full
#SBATCH --output=/home/zs377/logs/rebuild_full_%j.out
#SBATCH --error=/home/zs377/logs/rebuild_full_%j.err
#SBATCH --time=23:00:00
#SBATCH --partition=devel
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

source /etc/profile.d/modules.sh
module load miniconda
conda activate openpi

cd /home/zs377/project_pi_tkf6/zs377/projects/openpi_liberoplus

echo "[$(date)] Step 1: Delete old caches (force rebuild with 330 features)..."
rm -f /nfs/roberts/scratch/pi_tkf6/zs377/analysis_conformal_v2/feature_cache/*.pkl

echo "[$(date)] Step 2: conformal_analysis_v2.py — rebuild all 3 group caches..."
python scripts/attention_analysis/conformal_analysis_v2.py 2>&1

echo "[$(date)] Step 3: comprehensive_window_analysis.py — windowed AUROC over 330 features..."
python scripts/attention_analysis/comprehensive_window_analysis.py 2>&1

echo "[$(date)] Step 4: phase_aware_analysis.py — phase ranking + chunked conformal..."
python scripts/attention_analysis/phase_aware_analysis.py 2>&1

echo "[$(date)] Step 5: token_aggregation_analysis.py — token aggregation comparison..."
python scripts/attention_analysis/token_aggregation_analysis.py 2>&1

echo "[$(date)] All done."
