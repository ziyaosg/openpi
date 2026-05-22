#!/bin/bash
#SBATCH --job-name=phase-analysis
#SBATCH --output=/home/zs377/logs/rebuild_step2_%j.out
#SBATCH --error=/home/zs377/logs/rebuild_step2_%j.err
#SBATCH --time=5:30:00
#SBATCH --partition=devel
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

source /etc/profile.d/modules.sh
module load miniconda
conda activate openpi

cd /home/zs377/project_pi_tkf6/zs377/projects/openpi_liberoplus

echo "[$(date)] comprehensive_window_analysis.py..."
python scripts/attention_analysis/comprehensive_window_analysis.py 2>&1

echo "[$(date)] phase_aware_analysis.py..."
python scripts/attention_analysis/phase_aware_analysis.py 2>&1

echo "[$(date)] token_aggregation_analysis.py..."
python scripts/attention_analysis/token_aggregation_analysis.py 2>&1

echo "[$(date)] Step 2 done."
